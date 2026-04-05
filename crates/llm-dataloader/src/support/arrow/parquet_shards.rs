use std::{
    fs::File,
    path::PathBuf,
};

use arrow::{
    array::RecordBatch,
    error::Result as ArrowResult,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

pub fn read_parquet_shards(paths: Vec<PathBuf>) -> impl Iterator<Item = ArrowResult<RecordBatch>> {
    paths
        .into_iter()
        .map(|path| {
            let file = File::open(&path)?;
            ParquetRecordBatchReaderBuilder::try_new(file).and_then(|b| b.build())
        })
        .flat_map(|res| {
            let iter: Box<dyn Iterator<Item = ArrowResult<RecordBatch>>> = match res {
                Err(e) => Box::new(std::iter::once(Err(e.into()))),
                Ok(reader) => Box::new(reader),
            };
            iter
        })
}
