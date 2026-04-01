#![allow(unused)]

use arrow::array::{RecordBatch, StringArray};
use burn::serde::Serialize;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use rand::prelude::SliceRandom;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;
use wordchipper::support::slices::inner_str_view;
use wordchipper::{TokenEncoder, TokenType, Tokenizer};

pub struct TokenBatch<T: TokenType> {
    pub batch: Vec<Vec<T>>,
}

impl<T: TokenType> TokenBatch<T> {
    pub fn total_tokens(&self) -> usize {
        self.batch.iter().map(|x| x.len()).sum()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBatchOptions {
    /// The number of sequences to load per batch.
    pub batch_size: usize,

    /// The maximum number of tokens in a sequence.
    pub batch_seq_len: usize,

    /// The minimum number of sequences to keep in the buffer
    /// before loading more sequences.
    pub min_buffer: usize,
}

impl Default for TokenBatchOptions {
    fn default() -> Self {
        Self {
            batch_size: 32,
            batch_seq_len: 2048,
            min_buffer: 1024,
        }
    }
}

pub struct TokenBatchLoader<T: TokenType> {
    tokenizer: Arc<Tokenizer<T>>,
    options: TokenBatchOptions,
    bos_token: T,

    files: Vec<PathBuf>,
}

impl<T: TokenType> TokenBatchLoader<T> {
    pub fn new(
        tokenizer: Arc<Tokenizer<T>>,
        files: Vec<PathBuf>,
        options: TokenBatchOptions,
        bos_token: T,
    ) -> Self {
        Self {
            tokenizer,
            options,
            bos_token,
            files,
        }
    }

    pub fn iter(
        &self,
        shuffle: bool,
    ) -> TokenBatchIterator<T> {
        let mut files = self.files.clone();
        if shuffle {
            files.shuffle(&mut rand::rng());
        }

        TokenBatchIterator::new(
            self.tokenizer.clone(),
            files,
            self.options.clone(),
            self.bos_token,
        )
    }
}

pub struct TokenBatchIterator<T: TokenType> {
    tokenizer: Arc<Tokenizer<T>>,
    options: TokenBatchOptions,
    bos_token: T,

    files: Vec<PathBuf>,
    buffer: Vec<Vec<T>>,
    reader: Option<ParquetRecordBatchReader>,
}

impl<T: TokenType> TokenBatchIterator<T> {
    pub fn new(
        tokenizer: Arc<Tokenizer<T>>,
        files: Vec<PathBuf>,
        options: TokenBatchOptions,
        bos_token: T,
    ) -> Self {
        Self {
            tokenizer,
            options,
            bos_token,
            files,
            buffer: Vec::new(),
            reader: None,
        }
    }
}

impl<T: TokenType> Iterator for TokenBatchIterator<T> {
    type Item = Result<TokenBatch<T>, Box<dyn std::error::Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_batch() {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

impl<T: TokenType> TokenBatchIterator<T> {
    fn next_record_batch(&mut self) -> Result<Option<RecordBatch>, Box<dyn std::error::Error>> {
        loop {
            if self.reader.is_none() {
                if self.files.is_empty() {
                    return Ok(None);
                }
                let path = self.files.remove(0);
                let file = std::fs::File::open(path)?;
                let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
                self.reader = Some(reader);
            }

            if let Some(res) = self.reader.as_mut().unwrap().next() {
                return Ok(Some(res?));
            } else {
                self.reader = None;
                continue;
            }
        }
    }

    fn refill_buffer(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        while self.buffer.len() < self.options.min_buffer {
            if let Some(record_batch) = self.next_record_batch()? {
                let column = record_batch
                    .column_by_name("text")
                    .expect("failed to find 'text' column in batch")
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();

                let samples: Vec<String> = column
                    .into_iter()
                    .flat_map(|x| x.map(|s| s.to_string()))
                    .collect::<Vec<String>>();

                let tokens = self.tokenizer.try_encode_batch(&inner_str_view(&samples))?;

                self.buffer.extend(tokens);
            } else {
                break;
            }
        }

        Ok(())
    }

    fn next_batch(&mut self) -> Result<Option<TokenBatch<T>>, Box<dyn std::error::Error>> {
        let mut batch = Vec::with_capacity(self.options.batch_size);

        let row_capacity = self.options.batch_seq_len;
        while batch.len() < self.options.batch_size {
            let mut row: Vec<T> = Vec::with_capacity(row_capacity);

            while row.len() < row_capacity {
                self.refill_buffer()?;
                if self.buffer.is_empty() {
                    break;
                }

                row.push(self.bos_token);
                let remaining = row_capacity - row.len();

                let idx = if let Some((idx, _)) = self
                    .buffer
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, ts)| {
                        let k = ts.len();
                        if k <= remaining { Some((idx, k)) } else { None }
                    })
                    .max_by_key(|(_, k)| *k)
                {
                    // Find and add the longest sequence in the buffer that fits entirely in the row.
                    idx
                } else {
                    // No doc fits entirely in the row.
                    // Find the shortest sequence to crop, to minimize waste.
                    let (idx, _) = self
                        .buffer
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, ts)| ts.len())
                        .unwrap();
                    idx
                };

                let sequence = self.buffer.remove(idx);
                if sequence.len() > row_capacity - row.len() {
                    row.extend(&sequence[..(row_capacity - row.len())]);
                } else {
                    row.extend(sequence);
                }
            }

            if row.is_empty() {
                break;
            }

            batch.push(row);
        }

        if batch.is_empty() {
            Ok(None)
        } else {
            Ok(Some(TokenBatch { batch }))
        }
    }
}
