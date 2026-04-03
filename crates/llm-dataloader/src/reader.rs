use std::{
    path::PathBuf,
    sync::Arc,
};

use arrow::{
    array::{
        Array,
        ArrayRef,
        ListBuilder,
        RecordBatch,
        StringArray,
        UInt32Builder,
    },
    datatypes::{
        DataType,
        Field,
        Schema,
    },
    error::{
        ArrowError,
        Result as ArrowResult,
    },
};
use wordchipper::{
    TokenEncoder,
    Tokenizer,
    support::slices::inner_str_view,
};

use crate::support::arrow::parquet_shards::ParquetShardsBatchReader;

pub fn read_tokenized_batches(
    shard_paths: Vec<PathBuf>,
    tokenizer: Arc<Tokenizer<u32>>,
) -> impl Iterator<Item = ArrowResult<RecordBatch>> {
    let tokens_field = Arc::new(Field::new(
        "tokens",
        DataType::List(Arc::new(Field::new_list_field(DataType::UInt32, false))),
        false,
    ));
    let schema = Arc::new(Schema::new(vec![
        Field::new("text", DataType::Utf8, false),
        tokens_field.as_ref().clone(),
    ]));

    ParquetShardsBatchReader::new(shard_paths).map(move |res| -> ArrowResult<RecordBatch> {
        let record_batch = res?;

        let text_column = record_batch
            .column_by_name("text")
            .expect("failed to find 'text' column in batch")
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        let samples: Vec<String> = text_column
            .into_iter()
            .flat_map(|x| x.map(|s| s.to_string()))
            .collect::<Vec<String>>();

        let tokens = tokenizer
            .try_encode_batch(&inner_str_view(&samples))
            .map_err(|e| ArrowError::ComputeError(e.to_string()))?;

        let mut tokens_column = ListBuilder::new(UInt32Builder::new())
            .with_field(Arc::new(Field::new_list_field(DataType::UInt32, false)));
        for ts in tokens {
            tokens_column.values().append_slice(&ts);
            tokens_column.append(true);
        }

        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(text_column.clone()) as ArrayRef,
                Arc::new(tokens_column.finish()) as ArrayRef,
            ],
        )
    })
}
