use std::{
    fs::File,
    path::PathBuf,
};

use arrow::{
    array::RecordBatch,
    error::Result as ArrowResult,
};
use parquet::arrow::arrow_reader::{
    ParquetRecordBatchReader,
    ParquetRecordBatchReaderBuilder,
};

pub struct ParquetShardsCollection {
    paths: Vec<PathBuf>,
}

impl ParquetShardsCollection {
    pub fn new(paths: Vec<PathBuf>) -> Self {
        Self { paths }
    }

    pub fn len(&self) -> usize {
        self.paths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    pub fn batch_reader(&mut self) -> ParquetShardsBatchReader {
        ParquetShardsBatchReader::new(self.paths.clone())
    }
}

pub struct ParquetShardsBatchReader {
    paths: Vec<PathBuf>,
    current: Option<ParquetRecordBatchReader>,
}

impl ParquetShardsBatchReader {
    pub fn new(paths: Vec<PathBuf>) -> Self {
        Self {
            paths,
            current: None,
        }
    }
}

impl Iterator for ParquetShardsBatchReader {
    type Item = ArrowResult<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current.is_none() {
                let path = self.paths.pop()?;
                let file = match File::open(&path) {
                    Ok(f) => f,
                    Err(e) => return Some(Err(e.into())),
                };

                let reader =
                    match ParquetRecordBatchReaderBuilder::try_new(file).and_then(|b| b.build()) {
                        Ok(r) => r,
                        Err(e) => return Some(Err(e.into())),
                    };

                self.current = Some(reader);
            }

            if let Some(reader) = self.current.as_mut() {
                match reader.next() {
                    Some(Ok(batch)) => return Some(Ok(batch)),
                    Some(Err(e)) => return Some(Err(e)),
                    None => {
                        self.current = None;
                        continue;
                    }
                }
            }
        }
    }
}
