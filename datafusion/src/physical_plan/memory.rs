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

//! Execution plan for reading in-memory batches of data

use core::fmt;
use std::any::Any;
use std::sync::Arc;
use std::task::{Context, Poll};

use super::{
    common, DisplayFormatType, ExecutionPlan, Partitioning, RecordBatchStream,
    SendableRecordBatchStream, Statistics,
};
use crate::error::{DataFusionError, Result};
use crate::physical_plan::{ConsumeStatus, Consumer};
use arrow::datatypes::{Field, Schema, SchemaRef};
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;

use async_trait::async_trait;
use futures::Stream;

/// Execution plan for reading in-memory batches of data
pub struct MemoryExec {
    /// The partitions to query
    partitions: Vec<Vec<RecordBatch>>,
    /// Schema representing the data before projection
    schema: SchemaRef,
    /// Schema representing the data after the optional projection is applied
    projected_schema: SchemaRef,
    /// Optional projection
    projection: Option<Vec<usize>>,
}

impl fmt::Debug for MemoryExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "partitions: [...]")?;
        write!(f, "schema: {:?}", self.projected_schema)?;
        write!(f, "projection: {:?}", self.projection)
    }
}

impl MemoryExec {
    /// Create a new execution plan for reading in-memory record batches
    /// The provided `schema` should not have the projection applied.
    pub fn try_new(
        partitions: Vec<Vec<RecordBatch>>,
        schema: SchemaRef,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        let projected_schema = match &projection {
            Some(columns) => {
                let fields: Result<Vec<Field>> = columns
                    .iter()
                    .map(|i| {
                        if *i < schema.fields().len() {
                            Ok(schema.field(*i).clone())
                        } else {
                            Err(DataFusionError::Internal(
                                "Projection index out of range".to_string(),
                            ))
                        }
                    })
                    .collect();
                Arc::new(Schema::new(fields?))
            }
            None => Arc::clone(&schema),
        };
        Ok(Self {
            partitions: partitions,
            schema,
            projected_schema,
            projection,
        })
    }
}

#[async_trait]
impl ExecutionPlan for MemoryExec {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Get the schema for this execution plan
    fn schema(&self) -> SchemaRef {
        self.projected_schema.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        // this is a leaf node and has no children
        vec![]
    }

    /// Get the output partitioning of this plan
    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(self.partitions.len())
    }

    fn with_new_children(
        &self,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Err(DataFusionError::Internal(format!(
            "Children cannot be replaced in {:?}",
            self
        )))
    }

    async fn execute(&self, partition: usize, consumer: &mut dyn Consumer) -> Result<()> {
        let partition = &self.partitions[partition];
        // apply projection
        match &self.projection {
            Some(columns) => {
                for batch in partition {
                    let projected_batch = RecordBatch::try_new(
                        self.projected_schema.clone(),
                        columns.iter().map(|i| batch.column(*i).clone()).collect(),
                    )?;
                    if consumer.consume(projected_batch)? == ConsumeStatus::Terminate {
                        break;
                    }
                }
            }
            None => {
                for batch in partition {
                    if consumer.consume(batch.clone())? == ConsumeStatus::Terminate {
                        break;
                    }
                }
            }
        }
        consumer.finish()?;
        Ok(())
    }

    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                let partitions: Vec<_> =
                    self.partitions.iter().map(|b| b.len()).collect();
                write!(
                    f,
                    "MemoryExec: partitions={}, partition_sizes={:?}",
                    partitions.len(),
                    partitions
                )
            }
        }
    }

    /// We recompute the statistics dynamically from the arrow metadata as it is pretty cheap to do so
    fn statistics(&self) -> Statistics {
        common::compute_record_batch_statistics(
            self.partitions.iter().map(|v| v.as_slice()),
            &self.schema,
            self.projection.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical_plan::ColumnStatistics;
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use futures::StreamExt;

    fn mock_data() -> Result<(SchemaRef, RecordBatch)> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
            Field::new("c", DataType::Int32, false),
            Field::new("d", DataType::Int32, true),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Int32Array::from(vec![4, 5, 6])),
                Arc::new(Int32Array::from(vec![None, None, Some(9)])),
                Arc::new(Int32Array::from(vec![7, 8, 9])),
            ],
        )?;

        Ok((schema, batch))
    }

    #[tokio::test]
    async fn test_with_projection() -> Result<()> {
        let (schema, batch) = mock_data()?;

        let executor = MemoryExec::try_new(vec![vec![batch]], schema, Some(vec![2, 1]))?;
        let statistics = executor.statistics();

        assert_eq!(statistics.num_rows, Some(3));
        assert_eq!(
            statistics.column_statistics,
            Some(vec![
                ColumnStatistics {
                    null_count: Some(2),
                    max_value: None,
                    min_value: None,
                    distinct_count: None,
                },
                ColumnStatistics {
                    null_count: Some(0),
                    max_value: None,
                    min_value: None,
                    distinct_count: None,
                },
            ])
        );

        // scan with projection
        let mut batches = vec![];
        executor.execute(0, &mut batches).await?;
        let batch2 = &batches[0];
        assert_eq!(2, batch2.schema().fields().len());
        assert_eq!("c", batch2.schema().field(0).name());
        assert_eq!("b", batch2.schema().field(1).name());
        assert_eq!(2, batch2.num_columns());

        Ok(())
    }

    #[tokio::test]
    async fn test_without_projection() -> Result<()> {
        let (schema, batch) = mock_data()?;

        let executor = MemoryExec::try_new(vec![vec![batch]], schema, None)?;
        let statistics = executor.statistics();

        assert_eq!(statistics.num_rows, Some(3));
        assert_eq!(
            statistics.column_statistics,
            Some(vec![
                ColumnStatistics {
                    null_count: Some(0),
                    max_value: None,
                    min_value: None,
                    distinct_count: None,
                },
                ColumnStatistics {
                    null_count: Some(0),
                    max_value: None,
                    min_value: None,
                    distinct_count: None,
                },
                ColumnStatistics {
                    null_count: Some(2),
                    max_value: None,
                    min_value: None,
                    distinct_count: None,
                },
                ColumnStatistics {
                    null_count: Some(0),
                    max_value: None,
                    min_value: None,
                    distinct_count: None,
                },
            ])
        );

        let mut batches = vec![];
        executor.execute(0, &mut batches).await?;
        let batch1 = &batches[0];
        assert_eq!(4, batch1.schema().fields().len());
        assert_eq!(4, batch1.num_columns());

        Ok(())
    }
}
