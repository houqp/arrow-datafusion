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

//! CoalesceBatchesExec combines small batches into larger batches for more efficient use of
//! vectorized processing by upstream operators.

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::error::{DataFusionError, Result};
use crate::physical_plan::{ConsumeStatus, Consumer};
use crate::physical_plan::{
    DisplayFormatType, ExecutionPlan, Partitioning, RecordBatchStream,
    SendableRecordBatchStream,
};

use arrow::compute::kernels::concat::concat;
use arrow::datatypes::SchemaRef;
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use futures::stream::{Stream, StreamExt};
use log::debug;

use super::metrics::{BaselineMetrics, MetricsSet};
use super::{metrics::ExecutionPlanMetricsSet, Statistics};

/// CoalesceBatchesExec combines small batches into larger batches for more efficient use of
/// vectorized processing by upstream operators.
#[derive(Debug)]
pub struct CoalesceBatchesExec {
    /// The input plan
    input: Arc<dyn ExecutionPlan>,
    /// Minimum number of rows for coalesces batches
    target_batch_size: usize,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
}

impl CoalesceBatchesExec {
    /// Create a new CoalesceBatchesExec
    pub fn new(input: Arc<dyn ExecutionPlan>, target_batch_size: usize) -> Self {
        Self {
            input,
            target_batch_size,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// The input plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Minimum number of rows for coalesces batches
    pub fn target_batch_size(&self) -> usize {
        self.target_batch_size
    }
}

#[async_trait]
impl ExecutionPlan for CoalesceBatchesExec {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Get the schema for this execution plan
    fn schema(&self) -> SchemaRef {
        // The coalesce batches operator does not make any changes to the schema of its input
        self.input.schema()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    /// Get the output partitioning of this plan
    fn output_partitioning(&self) -> Partitioning {
        // The coalesce batches operator does not make any changes to the partitioning of its input
        self.input.output_partitioning()
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match children.len() {
            1 => Ok(Arc::new(CoalesceBatchesExec::new(
                children[0].clone(),
                self.target_batch_size,
            ))),
            _ => Err(DataFusionError::Internal(
                "CoalesceBatchesExec wrong number of children".to_string(),
            )),
        }
    }

    async fn execute(&self, partition: usize, consumer: &mut dyn Consumer) -> Result<()> {
        let mut coalescer = BatchCoalescer::new(
            partition,
            self.input.schema(),
            self.target_batch_size,
            &self.metrics,
            consumer,
        );
        self.input.execute(partition, &mut coalescer).await
    }

    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                write!(
                    f,
                    "CoalesceBatchesExec: target_batch_size={}",
                    self.target_batch_size
                )
            }
        }
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Statistics {
        self.input.statistics()
    }
}

#[derive(Debug)]
struct BatchCoalescer<'a> {
    /// The input schema
    schema: SchemaRef,
    /// Minimum number of rows for coalesces batches
    target_batch_size: usize,
    /// Buffered batches
    buffer: Vec<RecordBatch>,
    /// Buffered row count
    buffered_rows: usize,
    consumer: &'a mut dyn Consumer,
}

impl<'a> BatchCoalescer<'a> {
    fn new(
        partition: usize,
        schema: SchemaRef,
        target_batch_size: usize,
        metrics: &ExecutionPlanMetricsSet,
        consumer: &'a mut dyn Consumer,
    ) -> Self {
        Self {
            schema,
            target_batch_size,
            buffer: Vec::new(),
            buffered_rows: 0,
            // baseline_metrics: BaselineMetrics::new(&metrics, partition),
            consumer,
        }
    }
}

#[async_trait]
impl<'a> Consumer for BatchCoalescer<'a> {
    // FIXME: add back timer
    // // Get a clone (uses same underlying atomic) as self gets borrowed below
    // let cloned_time = self.baseline_metrics.elapsed_compute().clone();
    // // records time on drop
    // let _timer = cloned_time.timer();

    async fn consume(&mut self, batch: RecordBatch) -> Result<ConsumeStatus> {
        if batch.num_rows() >= self.target_batch_size && self.buffer.is_empty() {
            self.consumer.consume(batch).await
        } else if batch.num_rows() == 0 {
            // discard empty batches
            Ok(ConsumeStatus::Continue)
        } else {
            // add to the buffered batches
            self.buffered_rows += batch.num_rows();
            self.buffer.push(batch);
            // check to see if we have enough batches yet
            if self.buffered_rows >= self.target_batch_size {
                // combine the batches and return
                let batch =
                    concat_batches(&self.schema, &self.buffer, self.buffered_rows)?;
                // reset buffer state
                self.buffer.clear();
                self.buffered_rows = 0;
                // push batch downstream
                self.consumer.consume(batch).await
            } else {
                Ok(ConsumeStatus::Continue)
            }
        }
    }

    async fn finish(&mut self) -> Result<()> {
        // we have reached the end of the input stream but there could still
        // be buffered batches
        if !self.buffer.is_empty() {
            // combine the batches and return
            let batch = concat_batches(&self.schema, &self.buffer, self.buffered_rows)?;
            // reset buffer state
            self.buffer.clear();
            self.buffered_rows = 0;
            // push batch downstream
            self.consumer.consume(batch).await?;
        }
        self.consumer.finish().await
    }
}

/// Concatenates an array of `RecordBatch` into one batch
pub fn concat_batches(
    schema: &SchemaRef,
    batches: &[RecordBatch],
    row_count: usize,
) -> ArrowResult<RecordBatch> {
    if batches.is_empty() {
        return Ok(RecordBatch::new_empty(schema.clone()));
    }
    let mut arrays = Vec::with_capacity(schema.fields().len());
    for i in 0..schema.fields().len() {
        let array = concat(
            &batches
                .iter()
                .map(|batch| batch.column(i).as_ref())
                .collect::<Vec<_>>(),
        )?;
        arrays.push(array);
    }
    debug!(
        "Combined {} batches containing {} rows",
        batches.len(),
        row_count
    );
    RecordBatch::try_new(schema.clone(), arrays)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical_plan::{memory::MemoryExec, repartition::RepartitionExec};
    use arrow::array::UInt32Array;
    use arrow::datatypes::{DataType, Field, Schema};

    #[tokio::test]
    async fn empty_record_batches() -> Result<()> {
        // downstream consumer's consume and finish methods need to be invoked even though all
        // record batches contain 0 rows

        let schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::UInt32, false)]));
        let empty_batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(UInt32Array::from(Vec::<u32>::new()))],
        )?;
        let plan = MemoryExec::try_new(vec![vec![empty_batch]], schema, None)?;
        let plan = CoalesceBatchesExec::new(Arc::new(plan), 1024);

        #[derive(Debug)]
        struct DummyConsumer {
            consume_called: bool,
            finish_called: bool,
        }

        #[async_trait]
        impl Consumer for DummyConsumer {
            async fn consume(&mut self, batch: RecordBatch) -> Result<ConsumeStatus> {
                self.consume_called = true;
                Ok(ConsumeStatus::Continue)
            }

            async fn finish(&mut self) -> Result<()> {
                self.finish_called = true;
                Ok(())
            }
        }

        let mut consumer = DummyConsumer {
            consume_called: false,
            finish_called: false,
        };
        plan.execute(0, &mut consumer).await?;

        assert_eq!(consumer.consume_called, true);
        assert_eq!(consumer.finish_called, true);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_concat_batches() -> Result<()> {
        let schema = test_schema();
        let partition = create_vec_batches(&schema, 10);
        let partitions = vec![partition];

        let output_partitions = coalesce_batches(&schema, partitions, 21).await?;
        assert_eq!(1, output_partitions.len());

        // input is 10 batches x 8 rows (80 rows)
        // expected output is batches of at least 20 rows (except for the final batch)
        let batches = &output_partitions[0];
        assert_eq!(4, batches.len());
        assert_eq!(24, batches[0].num_rows());
        assert_eq!(24, batches[1].num_rows());
        assert_eq!(24, batches[2].num_rows());
        assert_eq!(8, batches[3].num_rows());

        Ok(())
    }

    fn test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new("c0", DataType::UInt32, false)]))
    }

    fn create_vec_batches(schema: &Arc<Schema>, num_batches: usize) -> Vec<RecordBatch> {
        let batch = create_batch(schema);
        let mut vec = Vec::with_capacity(num_batches);
        for _ in 0..num_batches {
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

    async fn coalesce_batches(
        schema: &SchemaRef,
        input_partitions: Vec<Vec<RecordBatch>>,
        target_batch_size: usize,
    ) -> Result<Vec<Vec<RecordBatch>>> {
        // create physical plan
        let exec = MemoryExec::try_new(input_partitions, schema.clone(), None)?;
        let exec =
            RepartitionExec::try_new(Arc::new(exec), Partitioning::RoundRobinBatch(1))?;
        let exec: Arc<dyn ExecutionPlan> =
            Arc::new(CoalesceBatchesExec::new(Arc::new(exec), target_batch_size));

        // execute and collect results
        let output_partition_count = exec.output_partitioning().partition_count();
        let mut output_partitions = Vec::with_capacity(output_partition_count);
        for i in 0..output_partition_count {
            // execute this *output* partition and collect all batches
            let mut batches = vec![];
            exec.execute(i, &mut batches).await?;
            output_partitions.push(batches);
        }
        Ok(output_partitions)
    }
}
