// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! The repartition operator maps N input partitions to M output partitions based on a
//! partitioning scheme.

use std::cell::RefCell;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::{any::Any, vec};

use crate::error::{DataFusionError, Result};
use crate::physical_plan::hash_utils::create_hashes;
use crate::physical_plan::{ConsumeStatus, Consumer};
use crate::physical_plan::{DisplayFormatType, ExecutionPlan, Partitioning, Statistics};
use arrow::record_batch::RecordBatch;
use arrow::{array::Array, error::Result as ArrowResult};
use arrow::{compute::take, datatypes::SchemaRef};
use futures::stream::Stream;
use futures::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::common::{AbortOnDropMany, AbortOnDropSingle};
use super::metrics::{self, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet};
use async_trait::async_trait;

use hashbrown::HashMap;
use tokio::sync::{
    mpsc::{self, UnboundedReceiver, UnboundedSender},
    Mutex,
};
use tokio::task::JoinHandle;

/// Inner state of [`RepartitionExec`].
#[derive(Debug)]
struct RepartitionExecState {
    /// Channels for sending batches from input partitions to output partitions.
    /// Key is the partition number.
    channels: HashMap<
        usize,
        (
            UnboundedSender<Result<RecordBatch>>,
            UnboundedReceiver<Result<RecordBatch>>,
        ),
    >,

    /// Helper that ensures that that background job is killed once it is no longer needed.
    abort_helper: Arc<AbortOnDropMany<()>>,
}

/// The repartition operator maps N input partitions to M output partitions based on a
/// partitioning scheme. No guarantees are made about the order of the resulting partitions.
#[derive(Debug)]
pub struct RepartitionExec {
    /// Input execution plan
    input: Arc<dyn ExecutionPlan>,

    /// Partitioning scheme to use
    partitioning: Partitioning,

    /// Inner state that is initialized when the first output stream is created.
    state: Arc<Mutex<RepartitionExecState>>,

    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
}

#[derive(Debug, Clone)]
struct RepartitionMetrics {
    /// Time in nanos to execute child operator and fetch batches
    fetch_time: metrics::Time,
    /// Time in nanos to perform repartitioning
    repart_time: metrics::Time,
    /// Time in nanos for sending resulting batches to channels
    send_time: metrics::Time,
}

impl RepartitionMetrics {
    pub fn new(
        output_partition: usize,
        input_partition: usize,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Self {
        let label = metrics::Label::new("inputPartition", input_partition.to_string());

        // Time in nanos to execute child operator and fetch batches
        let fetch_time = MetricBuilder::new(metrics)
            .with_label(label.clone())
            .subset_time("fetch_time", output_partition);

        // Time in nanos to perform repartitioning
        let repart_time = MetricBuilder::new(metrics)
            .with_label(label.clone())
            .subset_time("repart_time", output_partition);

        // Time in nanos for sending resulting batches to channels
        let send_time = MetricBuilder::new(metrics)
            .with_label(label)
            .subset_time("send_time", output_partition);

        Self {
            fetch_time,
            repart_time,
            send_time,
        }
    }
}

impl RepartitionExec {
    /// Input execution plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Partitioning scheme to use
    pub fn partitioning(&self) -> &Partitioning {
        &self.partitioning
    }
}

impl RepartitionExec {
    /// Create a new RepartitionExec
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        partitioning: Partitioning,
    ) -> Result<Self> {
        Ok(RepartitionExec {
            input,
            partitioning,
            state: Arc::new(Mutex::new(RepartitionExecState {
                channels: HashMap::new(),
                abort_helper: Arc::new(AbortOnDropMany::<()>(vec![])),
            })),
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

fn spawn_repartition_task(
    partitioning: Partitioning,
    partition_senders: HashMap<usize, UnboundedSender<Result<RecordBatch>>>,
    input: Arc<dyn ExecutionPlan>,
    input_partition: usize,
    r_metrics: RepartitionMetrics,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut input_repartitioner =
            Repartitioner::new(partitioning, partition_senders, r_metrics);
        let re = input
            .execute(input_partition, &mut input_repartitioner)
            .await;

        // propagate error to receiver
        if let Err(err) = re {
            // TODO: propagate errors concurrently using tokio::spawn
            for (_, sender) in input_repartitioner.partition_senders {
                // If send fails, plan being torn
                // down, no place to send the error
                sender
                    .send(Err(DataFusionError::Execution(err.to_string())))
                    .ok();
            }
        }
    })
}

#[async_trait]
impl ExecutionPlan for RepartitionExec {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Get the schema for this execution plan
    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match children.len() {
            1 => Ok(Arc::new(RepartitionExec::try_new(
                children[0].clone(),
                self.partitioning.clone(),
            )?)),
            _ => Err(DataFusionError::Internal(
                "RepartitionExec wrong number of children".to_string(),
            )),
        }
    }

    fn output_partitioning(&self) -> Partitioning {
        self.partitioning.clone()
    }

    async fn execute(&self, partition: usize, consumer: &mut dyn Consumer) -> Result<()> {
        let mut state = self.state.lock().await;

        // if this is the first partition to be invoked then we need to set up initial state
        // FIXME: state.channels goes to empty after we executed all partitions as well, this is a
        // bug
        if state.channels.is_empty() {
            let num_input_partitions = self.input.output_partitioning().partition_count();
            let num_output_partitions = self.partitioning.partition_count();

            for ouput_partition in 0..num_output_partitions {
                // Note that this operator uses unbounded channels to avoid deadlocks because
                // the output partitions can be read in any order and this could cause input
                // partitions to be blocked when sending data to output UnboundedReceivers that are not
                // being read yet. This may cause high memory usage if the next operator is
                // reading output partitions in order rather than concurrently. One workaround
                // for this would be to add spill-to-disk capabilities.
                let (sender, receiver) = mpsc::unbounded_channel::<Result<RecordBatch>>();
                state.channels.insert(ouput_partition, (sender, receiver));
            }

            let mut join_handles = Vec::with_capacity(num_input_partitions);
            for input_partition in 0..num_input_partitions {
                let output_partition_senders: HashMap<_, _> = state
                    .channels
                    .iter()
                    .map(|(partition, (tx, _rx))| (*partition, tx.clone()))
                    .collect();
                // TODO: this seems wrong, we are only recording metrics for partition 0?
                let r_metrics =
                    RepartitionMetrics::new(input_partition, partition, &self.metrics);
                join_handles.push(spawn_repartition_task(
                    self.partitioning.clone(),
                    output_partition_senders,
                    self.input.clone(),
                    input_partition,
                    r_metrics,
                ));
            }

            state.abort_helper = Arc::new(AbortOnDropMany(join_handles))
        };

        let mut stream =
            UnboundedReceiverStream::new(state.channels.remove(&partition).unwrap().1);
        let _drop_abort_helper = Arc::clone(&state.abort_helper);
        drop(state); // unlock state to unblock other partition consumers

        while let Some(re) = stream.next().await {
            let batch = re?;
            let status = consumer.consume(batch).await?;
            if status == ConsumeStatus::Terminate {
                break;
            }
        }
        consumer.finish().await
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                write!(f, "RepartitionExec: partitioning={:?}", self.partitioning)
            }
        }
    }

    fn statistics(&self) -> Statistics {
        self.input.statistics()
    }
}

#[derive(Debug)]
struct Repartitioner {
    partitioning: Partitioning,
    num_partitions: usize,
    random_state: ahash::RandomState,
    hashes_buf: Vec<u64>,
    // sender for each output partition
    partition_senders: HashMap<usize, UnboundedSender<Result<RecordBatch>>>,
    r_metrics: RepartitionMetrics,
    // TODO: only needed for RoundRobinBatch
    counter: usize,
}

impl Repartitioner {
    fn new(
        partitioning: Partitioning,
        partition_senders: HashMap<usize, UnboundedSender<Result<RecordBatch>>>,
        r_metrics: RepartitionMetrics,
    ) -> Self {
        Self {
            partitioning,
            num_partitions: partition_senders.len(),
            partition_senders,
            r_metrics,
            // Use fixed random state
            random_state: ahash::RandomState::with_seeds(0, 0, 0, 0),
            counter: 0,
            hashes_buf: vec![],
        }
    }
}

#[async_trait]
impl Consumer for Repartitioner {
    async fn consume(&mut self, batch: RecordBatch) -> Result<ConsumeStatus> {
        // FIXME: time fetch
        // // fetch the next batch
        // let timer = r_metrics.fetch_time.timer();
        // let result = stream.next().await;
        // timer.done();

        // TODO: avoid matching on partitioning on each batch
        match &self.partitioning {
            Partitioning::RoundRobinBatch(_) => {
                let timer = self.r_metrics.send_time.timer();
                let output_partition = self.counter % self.num_partitions;
                if let Some(sender) = self.partition_senders.get_mut(&output_partition) {
                    if sender.send(Ok(batch)).is_err() {
                        // If the other end has hung up, it was an early shutdown (e.g. LIMIT)
                        self.partition_senders.remove(&output_partition);
                    }
                }
                self.counter += 1;
                timer.done();
            }
            Partitioning::Hash(exprs, _) => {
                // FIXME: propagate error through sender channels
                let timer = self.r_metrics.repart_time.timer();
                let arrays = exprs
                    .iter()
                    .map(|expr| Ok(expr.evaluate(&batch)?.into_array(batch.num_rows())))
                    .collect::<Result<Vec<_>>>()?;
                self.hashes_buf.clear();
                self.hashes_buf.resize(arrays[0].len(), 0);
                // Hash arrays and compute buckets based on number of partitions
                let hashes =
                    create_hashes(&arrays, &self.random_state, &mut self.hashes_buf)?;
                let num_partitions = self.num_partitions;
                let mut indices = vec![vec![]; num_partitions];
                for (index, hash) in hashes.iter().enumerate() {
                    indices[(*hash % num_partitions as u64) as usize].push(index as u64)
                }
                timer.done();

                for (output_partition, partition_indices) in
                    indices.into_iter().enumerate()
                {
                    if partition_indices.is_empty() {
                        continue;
                    }

                    let timer = self.r_metrics.repart_time.timer();
                    let indices = partition_indices.into();
                    // Produce batches based on indices
                    let columns = batch
                        .columns()
                        .iter()
                        .map(|c| {
                            take(c.as_ref(), &indices, None)
                                .map_err(|e| DataFusionError::Execution(e.to_string()))
                        })
                        .collect::<Result<Vec<Arc<dyn Array>>>>()?;
                    let output_batch = RecordBatch::try_new(batch.schema(), columns)?;
                    timer.done();

                    let timer = self.r_metrics.send_time.timer();
                    // if there is still a receiver, send to it
                    if let Some(tx) = self.partition_senders.get_mut(&output_partition) {
                        if tx.send(Ok(output_batch)).is_err() {
                            // If the other end has hung up, it was an early shutdown (e.g. LIMIT)
                            self.partition_senders.remove(&output_partition);
                        }
                    }
                    timer.done();
                }
            }
            other => {
                // this should be unreachable as long as the validation logic
                // in the constructor is kept up-to-date
                return Err(DataFusionError::NotImplemented(format!(
                    "Unsupported repartitioning scheme {:?}",
                    other
                )));
            }
        }

        Ok(ConsumeStatus::Continue)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::{
        assert_batches_sorted_eq,
        physical_plan::{collect, expressions::col, memory::MemoryExec},
        test::{
            assert_is_pending,
            exec::{
                assert_strong_count_converges_to_zero, BarrierExec, BlockingExec,
                ErrorExec, MockExec,
            },
        },
    };
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use arrow::{
        array::{ArrayRef, StringArray, UInt32Array},
        error::ArrowError,
    };
    use futures::FutureExt;

    #[tokio::test]
    async fn one_to_many_round_robin() -> Result<()> {
        // define input partitions
        let schema = test_schema();
        let partition = create_vec_batches(&schema, 50);
        let partitions = vec![partition];

        // repartition from 1 input to 4 output
        let output_partitions =
            repartition(&schema, partitions, Partitioning::RoundRobinBatch(4)).await?;

        assert_eq!(4, output_partitions.len());
        assert_eq!(13, output_partitions[0].len());
        assert_eq!(13, output_partitions[1].len());
        assert_eq!(12, output_partitions[2].len());
        assert_eq!(12, output_partitions[3].len());

        Ok(())
    }

    #[tokio::test]
    async fn many_to_one_round_robin() -> Result<()> {
        // define input partitions
        let schema = test_schema();
        let partition = create_vec_batches(&schema, 50);
        let partitions = vec![partition.clone(), partition.clone(), partition.clone()];

        // repartition from 3 input to 1 output
        let output_partitions =
            repartition(&schema, partitions, Partitioning::RoundRobinBatch(1)).await?;

        assert_eq!(1, output_partitions.len());
        assert_eq!(150, output_partitions[0].len());

        Ok(())
    }

    #[tokio::test]
    async fn many_to_many_round_robin() -> Result<()> {
        // define input partitions
        let schema = test_schema();
        let partition = create_vec_batches(&schema, 50);
        let partitions = vec![partition.clone(), partition.clone(), partition.clone()];

        // repartition from 3 input to 5 output
        let output_partitions =
            repartition(&schema, partitions, Partitioning::RoundRobinBatch(5)).await?;

        assert_eq!(5, output_partitions.len());
        assert_eq!(30, output_partitions[0].len());
        assert_eq!(30, output_partitions[1].len());
        assert_eq!(30, output_partitions[2].len());
        assert_eq!(30, output_partitions[3].len());
        assert_eq!(30, output_partitions[4].len());

        Ok(())
    }

    #[tokio::test]
    async fn many_to_many_hash_partition() -> Result<()> {
        // define input partitions
        let schema = test_schema();
        let partition = create_vec_batches(&schema, 50);
        let partitions = vec![partition.clone(), partition.clone(), partition.clone()];

        let output_partitions = repartition(
            &schema,
            partitions,
            Partitioning::Hash(vec![col("c0", &schema)?], 8),
        )
        .await?;

        let total_rows: usize = output_partitions
            .iter()
            .map(|x| x.iter().map(|x| x.num_rows()).sum::<usize>())
            .sum();

        assert_eq!(8, output_partitions.len());
        assert_eq!(total_rows, 8 * 50 * 3);

        Ok(())
    }

    fn test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new("c0", DataType::UInt32, false)]))
    }

    fn create_vec_batches(schema: &Arc<Schema>, n: usize) -> Vec<RecordBatch> {
        let batch = create_batch(schema);
        let mut vec = Vec::with_capacity(n);
        for _ in 0..n {
            vec.push(batch.clone());
        }
        vec
    }

    fn create_batch(schema: &Arc<Schema>) -> RecordBatch {
        RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(UInt32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]))],
        )
        .unwrap()
    }

    async fn repartition(
        schema: &SchemaRef,
        input_partitions: Vec<Vec<RecordBatch>>,
        partitioning: Partitioning,
    ) -> Result<Vec<Vec<RecordBatch>>> {
        // create physical plan
        let exec = MemoryExec::try_new(input_partitions, schema.clone(), None)?;
        let exec = RepartitionExec::try_new(Arc::new(exec), partitioning)?;

        // execute and collect results
        let mut output_partitions = vec![];
        for i in 0..exec.partitioning.partition_count() {
            // execute this *output* partition and collect all batches
            let mut batches = vec![];
            exec.execute(i, &mut batches).await?;
            output_partitions.push(batches);
        }
        Ok(output_partitions)
    }

    #[tokio::test]
    async fn many_to_many_round_robin_within_tokio_task() -> Result<()> {
        let join_handle: JoinHandle<Result<Vec<Vec<RecordBatch>>>> =
            tokio::spawn(async move {
                // define input partitions
                let schema = test_schema();
                let partition = create_vec_batches(&schema, 50);
                let partitions =
                    vec![partition.clone(), partition.clone(), partition.clone()];

                // repartition from 3 input to 5 output
                repartition(&schema, partitions, Partitioning::RoundRobinBatch(5)).await
            });

        let output_partitions = join_handle
            .await
            .map_err(|e| DataFusionError::Internal(e.to_string()))??;

        assert_eq!(5, output_partitions.len());
        assert_eq!(30, output_partitions[0].len());
        assert_eq!(30, output_partitions[1].len());
        assert_eq!(30, output_partitions[2].len());
        assert_eq!(30, output_partitions[3].len());
        assert_eq!(30, output_partitions[4].len());

        Ok(())
    }

    #[tokio::test]
    async fn unsupported_partitioning() {
        // have to send at least one batch through to provoke error
        let batch = RecordBatch::try_from_iter(vec![(
            "my_awesome_field",
            Arc::new(StringArray::from(vec!["foo", "bar"])) as ArrayRef,
        )])
        .unwrap();

        let schema = batch.schema();
        let input = MockExec::new(vec![Ok(batch)], schema);
        // This generates an error (partitioning type not supported)
        // but only after the plan is executed. The error should be
        // returned and no results produced
        let partitioning = Partitioning::UnknownPartitioning(1);
        let exec = RepartitionExec::try_new(Arc::new(input), partitioning).unwrap();
        let result = exec.execute(0, &mut Vec::new()).await;

        // Expect that an error is returned
        let result_string = result.unwrap_err().to_string();
        assert!(
            result_string
                .contains("Unsupported repartitioning scheme UnknownPartitioning(1)"),
            "actual: {}",
            result_string
        );
    }

    #[tokio::test]
    async fn error_for_input_exec() {
        // This generates an error on a call to execute. The error
        // should be returned and no results produced.

        let input = ErrorExec::new();
        let partitioning = Partitioning::RoundRobinBatch(1);
        let exec = RepartitionExec::try_new(Arc::new(input), partitioning).unwrap();

        // Note: this should pass (the stream can be created) but the
        // error when the input is executed should get passed back
        let result = exec.execute(0, &mut Vec::new()).await;

        // Expect that an error is returned
        let result_string = result.unwrap_err().to_string();
        assert!(
            result_string.contains("ErrorExec, unsurprisingly, errored in partition 0"),
            "actual: {}",
            result_string
        );
    }

    #[tokio::test]
    async fn repartition_with_error_in_stream() {
        let batch = RecordBatch::try_from_iter(vec![(
            "my_awesome_field",
            Arc::new(StringArray::from(vec!["foo", "bar"])) as ArrayRef,
        )])
        .unwrap();

        // input stream returns one good batch and then one error. The
        // error should be returned.
        let err = Err(ArrowError::ComputeError("bad data error".to_string()));

        let schema = batch.schema();
        let input = MockExec::new(vec![Ok(batch), err], schema);
        let partitioning = Partitioning::RoundRobinBatch(1);
        let exec = RepartitionExec::try_new(Arc::new(input), partitioning).unwrap();

        // Note: this should pass (the stream can be created) but the
        // error when the input is executed should get passed back
        let result = exec.execute(0, &mut Vec::new()).await;

        // Expect that an error is returned
        let result_string = result.unwrap_err().to_string();
        assert!(
            result_string.contains("bad data error"),
            "actual: {}",
            result_string
        );
    }

    #[tokio::test]
    async fn repartition_with_delayed_stream() {
        let batch1 = RecordBatch::try_from_iter(vec![(
            "my_awesome_field",
            Arc::new(StringArray::from(vec!["foo", "bar"])) as ArrayRef,
        )])
        .unwrap();

        let batch2 = RecordBatch::try_from_iter(vec![(
            "my_awesome_field",
            Arc::new(StringArray::from(vec!["frob", "baz"])) as ArrayRef,
        )])
        .unwrap();

        // The mock exec doesn't return immediately (instead it
        // requires the input to wait at least once)
        let schema = batch1.schema();
        let expected_batches = vec![batch1.clone(), batch2.clone()];
        let input = MockExec::new(vec![Ok(batch1), Ok(batch2)], schema);
        let partitioning = Partitioning::RoundRobinBatch(1);

        let exec = RepartitionExec::try_new(Arc::new(input), partitioning).unwrap();

        let expected = vec![
            "+------------------+",
            "| my_awesome_field |",
            "+------------------+",
            "| foo              |",
            "| bar              |",
            "| frob             |",
            "| baz              |",
            "+------------------+",
        ];

        assert_batches_sorted_eq!(&expected, &expected_batches);

        let mut batches = Vec::new();
        exec.execute(0, &mut batches).await.unwrap();
        assert_batches_sorted_eq!(&expected, &batches);
    }

    // #[tokio::test]
    // async fn robin_repartition_with_dropping_output_stream() {
    //     let partitioning = Partitioning::RoundRobinBatch(2);
    //     // The barrier exec waits to be pinged
    //     // requires the input to wait at least once)
    //     let input = Arc::new(make_barrier_exec());
    //
    //     // partition into two output streams
    //     let exec = RepartitionExec::try_new(input.clone(), partitioning).unwrap();
    //
    //     let output_stream0 = exec.execute(0).await.unwrap();
    //     let mut batches = Vec::new();
    //     // output stream 1 should *not* error and have one of the input batches
    //     let exec.execute(1, &mut batches).await.unwrap();
    //
    //     // now, purposely drop output stream 0
    //     // *before* any outputs are produced
    //     std::mem::drop(output_stream0);
    //
    //     // Now, start sending input
    //     input.wait().await;
    //
    //     let expected = vec![
    //         "+------------------+",
    //         "| my_awesome_field |",
    //         "+------------------+",
    //         "| baz              |",
    //         "| frob             |",
    //         "| gaz              |",
    //         "| grob             |",
    //         "+------------------+",
    //     ];
    //
    //     assert_batches_sorted_eq!(&expected, &batches);
    // }

    // #[tokio::test]
    // // As the hash results might be different on different platforms or
    // // wiht different compilers, we will compare the same execution with
    // // and without droping the output stream.
    // async fn hash_repartition_with_dropping_output_stream() {
    //     let partitioning = Partitioning::Hash(
    //         vec![Arc::new(crate::physical_plan::expressions::Column::new(
    //             "my_awesome_field",
    //             0,
    //         ))],
    //         2,
    //     );
    //
    //     // We first collect the results without droping the output stream.
    //     let input = Arc::new(make_barrier_exec());
    //     let exec = RepartitionExec::try_new(input.clone(), partitioning.clone()).unwrap();
    //     let output_stream1 = exec.execute(1).await.unwrap();
    //     input.wait().await;
    //     let batches_without_drop = crate::physical_plan::common::collect(output_stream1)
    //         .await
    //         .unwrap();
    //
    //     // run some checks on the result
    //     let items_vec = str_batches_to_vec(&batches_without_drop);
    //     let items_set: HashSet<&str> = items_vec.iter().copied().collect();
    //     assert_eq!(items_vec.len(), items_set.len());
    //     let source_str_set: HashSet<&str> =
    //         (&["foo", "bar", "frob", "baz", "goo", "gar", "grob", "gaz"])
    //             .iter()
    //             .copied()
    //             .collect();
    //     assert_eq!(items_set.difference(&source_str_set).count(), 0);
    //
    //     // Now do the same but dropping the stream before waiting for the barrier
    //     let input = Arc::new(make_barrier_exec());
    //     let exec = RepartitionExec::try_new(input.clone(), partitioning).unwrap();
    //     let output_stream0 = exec.execute(0).await.unwrap();
    //     let output_stream1 = exec.execute(1).await.unwrap();
    //     // now, purposely drop output stream 0
    //     // *before* any outputs are produced
    //     std::mem::drop(output_stream0);
    //     input.wait().await;
    //     let batches_with_drop = crate::physical_plan::common::collect(output_stream1)
    //         .await
    //         .unwrap();
    //
    //     assert_eq!(batches_without_drop, batches_with_drop);
    // }
    //
    fn str_batches_to_vec(batches: &[RecordBatch]) -> Vec<&str> {
        batches
            .iter()
            .flat_map(|batch| {
                assert_eq!(batch.columns().len(), 1);
                let string_array = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("Unexpected type for repartitoned batch");

                string_array
                    .iter()
                    .map(|v| v.expect("Unexpected null"))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    /// Create a BarrierExec that returns two partitions of two batches each
    fn make_barrier_exec() -> BarrierExec {
        let batch1 = RecordBatch::try_from_iter(vec![(
            "my_awesome_field",
            Arc::new(StringArray::from(vec!["foo", "bar"])) as ArrayRef,
        )])
        .unwrap();

        let batch2 = RecordBatch::try_from_iter(vec![(
            "my_awesome_field",
            Arc::new(StringArray::from(vec!["frob", "baz"])) as ArrayRef,
        )])
        .unwrap();

        let batch3 = RecordBatch::try_from_iter(vec![(
            "my_awesome_field",
            Arc::new(StringArray::from(vec!["goo", "gar"])) as ArrayRef,
        )])
        .unwrap();

        let batch4 = RecordBatch::try_from_iter(vec![(
            "my_awesome_field",
            Arc::new(StringArray::from(vec!["grob", "gaz"])) as ArrayRef,
        )])
        .unwrap();

        // The barrier exec waits to be pinged
        // requires the input to wait at least once)
        let schema = batch1.schema();
        BarrierExec::new(vec![vec![batch1, batch2], vec![batch3, batch4]], schema)
    }

    #[tokio::test]
    async fn test_drop_cancel() -> Result<()> {
        let schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Float32, true)]));

        let blocking_exec = Arc::new(BlockingExec::new(Arc::clone(&schema), 2));
        let refs = blocking_exec.refs();
        let repartition_exec = Arc::new(RepartitionExec::try_new(
            blocking_exec,
            Partitioning::UnknownPartitioning(1),
        )?);

        let fut = collect(repartition_exec);
        let mut fut = fut.boxed();

        assert_is_pending(&mut fut);
        drop(fut);
        assert_strong_count_converges_to_zero(refs).await;

        Ok(())
    }

    #[tokio::test]
    async fn hash_repartition_avoid_empty_batch() -> Result<()> {
        let batch = RecordBatch::try_from_iter(vec![(
            "a",
            Arc::new(StringArray::from(vec!["foo"])) as ArrayRef,
        )])
        .unwrap();
        let partitioning = Partitioning::Hash(
            vec![Arc::new(crate::physical_plan::expressions::Column::new(
                "a", 0,
            ))],
            2,
        );
        let schema = batch.schema();
        let input = MockExec::new(vec![Ok(batch)], schema);
        let exec = RepartitionExec::try_new(Arc::new(input), partitioning).unwrap();
        let mut batch0 = Vec::new();
        exec.execute(0, &mut batch0).await.unwrap();
        let mut batch1 = Vec::new();
        exec.execute(1, &mut batch1).await.unwrap();
        assert!(batch0.is_empty() || batch1.is_empty());
        Ok(())
    }
}
