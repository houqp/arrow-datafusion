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

//! Defines the projection execution plan. A projection determines which columns or expressions
//! are returned from a query. The SQL statement `SELECT a, b, a+b FROM t1` is an example
//! of a projection on table `t1` where the expressions `a`, `b`, and `a+b` are the
//! projection expressions. `SELECT` without `FROM` will only evaluate expressions.

use std::any::Any;
use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::error::{DataFusionError, Result};
use crate::physical_plan::{
    ColumnStatistics, DisplayFormatType, ExecutionPlan, Partitioning, PhysicalExpr,
};
use crate::physical_plan::{ConsumeStatus, Consumer};
use arrow::datatypes::{Field, Schema, SchemaRef};
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;

use super::expressions::Column;
use super::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use super::{RecordBatchStream, SendableRecordBatchStream, Statistics};
use async_trait::async_trait;
use futures::stream::Stream;
use futures::stream::StreamExt;

/// Execution plan for a projection
#[derive(Debug)]
pub struct ProjectionExec {
    /// The projection expressions stored as tuples of (expression, output column name)
    expr: Vec<(Arc<dyn PhysicalExpr>, String)>,
    /// The schema once the projection has been applied to the input
    schema: SchemaRef,
    /// The input plan
    input: Arc<dyn ExecutionPlan>,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
}

impl ProjectionExec {
    /// Create a projection on an input
    pub fn try_new(
        expr: Vec<(Arc<dyn PhysicalExpr>, String)>,
        input: Arc<dyn ExecutionPlan>,
    ) -> Result<Self> {
        let input_schema = input.schema();

        let fields: Result<Vec<Field>> = expr
            .iter()
            .map(|(e, name)| {
                let mut field = Field::new(
                    name,
                    e.data_type(&input_schema)?,
                    e.nullable(&input_schema)?,
                );
                field.set_metadata(get_field_metadata(e, &input_schema));

                Ok(field)
            })
            .collect();

        let schema = Arc::new(Schema::new_with_metadata(
            fields?,
            input_schema.metadata().clone(),
        ));

        Ok(Self {
            expr,
            schema,
            input: input.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// The projection expressions stored as tuples of (expression, output column name)
    pub fn expr(&self) -> &[(Arc<dyn PhysicalExpr>, String)] {
        &self.expr
    }

    /// The input plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }
}

#[async_trait]
impl ExecutionPlan for ProjectionExec {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Get the schema for this execution plan
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    /// Get the output partitioning of this plan
    fn output_partitioning(&self) -> Partitioning {
        self.input.output_partitioning()
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match children.len() {
            1 => Ok(Arc::new(ProjectionExec::try_new(
                self.expr.clone(),
                children[0].clone(),
            )?)),
            _ => Err(DataFusionError::Internal(
                "ProjectionExec wrong number of children".to_string(),
            )),
        }
    }

    async fn execute(&self, partition: usize, consumer: &mut dyn Consumer) -> Result<()> {
        let mut projector = Projector {
            schema: self.schema.clone(),
            expr: self.expr.iter().map(|x| x.0.clone()).collect(),
            baseline_metrics: BaselineMetrics::new(&self.metrics, partition),
            consumer,
        };
        self.input.execute(partition, &mut projector).await
    }

    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                let expr: Vec<String> = self
                    .expr
                    .iter()
                    .map(|(e, alias)| {
                        let e = e.to_string();
                        if &e != alias {
                            format!("{} as {}", e, alias)
                        } else {
                            e
                        }
                    })
                    .collect();

                write!(f, "ProjectionExec: expr=[{}]", expr.join(", "))
            }
        }
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Statistics {
        stats_projection(
            self.input.statistics(),
            self.expr.iter().map(|(e, _)| Arc::clone(e)),
        )
    }
}

/// If e is a direct column reference, returns the field level
/// metadata for that field, if any. Otherwise returns None
fn get_field_metadata(
    e: &Arc<dyn PhysicalExpr>,
    input_schema: &Schema,
) -> Option<BTreeMap<String, String>> {
    let name = if let Some(column) = e.as_any().downcast_ref::<Column>() {
        column.name()
    } else {
        return None;
    };

    input_schema
        .field_with_name(name)
        .ok()
        .and_then(|f| f.metadata().as_ref().cloned())
}

fn stats_projection(
    stats: Statistics,
    exprs: impl Iterator<Item = Arc<dyn PhysicalExpr>>,
) -> Statistics {
    let column_statistics = stats.column_statistics.map(|input_col_stats| {
        exprs
            .map(|e| {
                if let Some(col) = e.as_any().downcast_ref::<Column>() {
                    input_col_stats[col.index()].clone()
                } else {
                    // TODO stats: estimate more statistics from expressions
                    // (expressions should compute their statistics themselves)
                    ColumnStatistics::default()
                }
            })
            .collect()
    });

    Statistics {
        is_exact: stats.is_exact,
        num_rows: stats.num_rows,
        column_statistics,
        // TODO stats: knowing the type of the new columns we can guess the output size
        total_byte_size: None,
    }
}

#[derive(Debug)]
struct Projector<'a> {
    consumer: &'a mut dyn Consumer,
    expr: Vec<Arc<dyn PhysicalExpr>>,
    schema: SchemaRef,
    /// runtime metrics recording
    baseline_metrics: BaselineMetrics,
}

impl<'a> Consumer for Projector<'a> {
    fn consume(&mut self, batch: RecordBatch) -> Result<ConsumeStatus> {
        let _timer = self.baseline_metrics.elapsed_compute().timer();
        let projected_batch = self
            .expr
            .iter()
            .map(|expr| expr.evaluate(&batch))
            .map(|r| r.map(|v| v.into_array(batch.num_rows())))
            .collect::<Result<Vec<_>>>()
            .map_or_else(
                |e| Err(DataFusionError::into_arrow_external_error(e)),
                |arrays| RecordBatch::try_new(self.schema.clone(), arrays),
            )?;

        self.consumer.consume(projected_batch)
    }

    fn finish(&mut self) -> Result<()> {
        self.consumer.finish()
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::datasource::object_store::local::LocalFileSystem;
    use crate::physical_plan::expressions::{self, col};
    use crate::physical_plan::file_format::{CsvExec, PhysicalPlanConfig};
    use crate::scalar::ScalarValue;
    use crate::test::{self};
    use crate::test_util;
    use futures::future;

    #[tokio::test]
    async fn project_first_column() -> Result<()> {
        let schema = test_util::aggr_test_schema();

        let partitions = 4;
        let (_, files) =
            test::create_partitioned_csv("aggregate_test_100.csv", partitions)?;

        let csv = CsvExec::new(
            PhysicalPlanConfig {
                object_store: Arc::new(LocalFileSystem {}),
                file_schema: Arc::clone(&schema),
                file_groups: files,
                statistics: Statistics::default(),
                projection: None,
                batch_size: 1024,
                limit: None,
                table_partition_cols: vec![],
            },
            true,
            b',',
        );

        // pick column c1 and name it column c1 in the output schema
        let projection = ProjectionExec::try_new(
            vec![(col("c1", &schema)?, "c1".to_string())],
            Arc::new(csv),
        )?;

        let col_field = projection.schema.field(0);
        let col_metadata = col_field.metadata().clone().unwrap().clone();
        let data: &str = &col_metadata["testing"];
        assert_eq!(data, "test");

        let mut partition_count = 0;
        let mut row_count = 0;
        for partition in 0..projection.output_partitioning().partition_count() {
            partition_count += 1;
            let mut batches = Vec::new();
            projection.execute(partition, &mut batches).await?;
            row_count += batches
                .iter()
                .map(|batch| {
                    assert_eq!(1, batch.num_columns());
                    batch.num_rows()
                })
                .fold(0, |acc, x| acc + x);
        }
        assert_eq!(partitions, partition_count);
        assert_eq!(100, row_count);

        Ok(())
    }

    #[tokio::test]
    async fn test_stats_projection_columns_only() {
        let source = Statistics {
            is_exact: true,
            num_rows: Some(5),
            total_byte_size: Some(23),
            column_statistics: Some(vec![
                ColumnStatistics {
                    distinct_count: Some(5),
                    max_value: Some(ScalarValue::Int64(Some(21))),
                    min_value: Some(ScalarValue::Int64(Some(-4))),
                    null_count: Some(0),
                },
                ColumnStatistics {
                    distinct_count: Some(1),
                    max_value: Some(ScalarValue::Utf8(Some(String::from("x")))),
                    min_value: Some(ScalarValue::Utf8(Some(String::from("a")))),
                    null_count: Some(3),
                },
                ColumnStatistics {
                    distinct_count: None,
                    max_value: Some(ScalarValue::Float32(Some(1.1))),
                    min_value: Some(ScalarValue::Float32(Some(0.1))),
                    null_count: None,
                },
            ]),
        };

        let exprs: Vec<Arc<dyn PhysicalExpr>> = vec![
            Arc::new(expressions::Column::new("col1", 1)),
            Arc::new(expressions::Column::new("col0", 0)),
        ];

        let result = stats_projection(source, exprs.into_iter());

        let expected = Statistics {
            is_exact: true,
            num_rows: Some(5),
            total_byte_size: None,
            column_statistics: Some(vec![
                ColumnStatistics {
                    distinct_count: Some(1),
                    max_value: Some(ScalarValue::Utf8(Some(String::from("x")))),
                    min_value: Some(ScalarValue::Utf8(Some(String::from("a")))),
                    null_count: Some(3),
                },
                ColumnStatistics {
                    distinct_count: Some(5),
                    max_value: Some(ScalarValue::Int64(Some(21))),
                    min_value: Some(ScalarValue::Int64(Some(-4))),
                    null_count: Some(0),
                },
            ]),
        };

        assert_eq!(result, expected);
    }
}
