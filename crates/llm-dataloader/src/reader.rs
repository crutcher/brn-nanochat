use std::sync::Arc;

use arrow::{
    array::{
        Array,
        RecordBatch,
        StringArray,
    },
    error::{
        ArrowError,
        Result as ArrowResult,
    },
};
use serde::{
    Deserialize,
    Serialize,
};
use wordchipper::{
    TokenEncoder,
    Tokenizer,
    support::slices::inner_str_view,
};

pub fn select_text_columns<I, S>(
    column: S,
    iter: I,
) -> impl Iterator<Item = ArrowResult<Vec<String>>>
where
    I: Iterator<Item = ArrowResult<RecordBatch>>,
    S: AsRef<str>,
{
    let column = column.as_ref().to_string();
    iter.map(move |res| -> ArrowResult<Vec<String>> {
        let record_batch = res?;

        let text_column = record_batch
            .column_by_name(&column)
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        let samples: Vec<String> = text_column
            .into_iter()
            .flat_map(|x| x.map(|s| s.to_string()))
            .collect::<Vec<String>>();

        Ok(samples)
    })
}

pub fn tokenize_text_batches<I>(
    tokenizer: Arc<Tokenizer<u32>>,
    iter: I,
) -> impl Iterator<Item = ArrowResult<Vec<Vec<u32>>>>
where
    I: Iterator<Item = ArrowResult<Vec<String>>>,
{
    iter.map(move |res| -> ArrowResult<Vec<Vec<u32>>> {
        let text_batch = res?;

        let tokens = tokenizer
            .try_encode_batch(&inner_str_view(&text_batch))
            .map_err(|e| ArrowError::ComputeError(e.to_string()))?;

        Ok(tokens)
    })
}

pub fn compact_dense_token_blocks<I>(
    options: TokenBatchIteratorOptions,
    bos_seq: Vec<u32>,
    eos_seq: Vec<u32>,
    iter: I,
) -> impl Iterator<Item = ArrowResult<Vec<Vec<u32>>>>
where
    I: Iterator<Item = ArrowResult<Vec<Vec<u32>>>>,
{
    DenseTokenBlockBatcher::new(iter, options, bos_seq, eos_seq)
}

pub struct DenseTokenBlockBatcher<I>
where
    I: Iterator<Item = ArrowResult<Vec<Vec<u32>>>>,
{
    iter: I,
    options: TokenBatchIteratorOptions,
    bos_seq: Vec<u32>,
    eos_seq: Vec<u32>,
    buffer: Vec<Vec<u32>>,
}

impl<I> DenseTokenBlockBatcher<I>
where
    I: Iterator<Item = ArrowResult<Vec<Vec<u32>>>>,
{
    pub fn new(
        iter: I,
        options: TokenBatchIteratorOptions,
        bos_seq: Vec<u32>,
        eos_seq: Vec<u32>,
    ) -> Self {
        Self {
            iter,
            options,
            buffer: Vec::new(),
            bos_seq,
            eos_seq,
        }
    }

    fn refill_buffer(&mut self) -> ArrowResult<()> {
        while self.buffer.len() < self.options.min_buffer {
            if let Some(res) = self.iter.next() {
                self.buffer.extend(res?);
            } else {
                break;
            }
        }
        Ok(())
    }

    fn next_batch(&mut self) -> ArrowResult<Option<Vec<Vec<u32>>>> {
        let mut batch = Vec::with_capacity(self.options.batch_size);

        let row_capacity = self.options.batch_seq_len;
        while batch.len() < self.options.batch_size {
            let mut row: Vec<u32> = Vec::with_capacity(row_capacity);

            while row.len() < row_capacity {
                self.refill_buffer()?;
                if self.buffer.is_empty() {
                    break;
                }

                row.extend(&self.bos_seq);

                let remaining = row_capacity - row.len();

                let mut best_fit: Option<(usize, usize)> = None; // (idx, length)
                let mut shortest: Option<(usize, usize)> = None; // (idx, length)

                for (i, ts) in self.buffer.iter().enumerate() {
                    let k = ts.len();
                    let this = (i, k);

                    // Option 1: The longest buffer sequence that fits entirely in the row.
                    if k <= remaining && (best_fit.is_none() || best_fit.unwrap().1 > k) {
                        best_fit = Some((i, k));
                    }

                    // Option 2: The shortest buffer sequence to crop, to minimize waste.
                    if shortest.is_none() || shortest.unwrap().1 > k {
                        shortest = Some(this);
                    }
                }

                let idx = best_fit.unwrap_or(shortest.unwrap()).0;

                row.extend(self.buffer.remove(idx));
                row.extend(&self.eos_seq);
                row.truncate(row_capacity);
            }

            if row.is_empty() {
                break;
            }

            batch.push(row);
        }

        if !batch.is_empty() {
            // Enforce the no-partial rows policy.
            if batch.last().unwrap().len() < self.options.batch_seq_len {
                batch.pop();
            }
        }

        if batch.len() != self.options.batch_size {
            // Enforce the exact batch size policy.
            return Ok(None);
        }

        if batch.is_empty() {
            Ok(None)
        } else {
            Ok(Some(batch))
        }
    }
}

impl<I> Iterator for DenseTokenBlockBatcher<I>
where
    I: Iterator<Item = ArrowResult<Vec<Vec<u32>>>>,
{
    type Item = ArrowResult<Vec<Vec<u32>>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_batch() {
            Ok(None) => None,
            Ok(Some(batch)) => Some(Ok(batch)),
            Err(err) => Some(Err(err)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBatchIteratorOptions {
    /// The number of sequences to load per batch.
    pub batch_size: usize,

    /// The maximum number of tokens in a sequence.
    pub batch_seq_len: usize,

    /// The minimum number of sequences to keep in the buffer
    /// before loading more sequences.
    pub min_buffer: usize,
}

impl Default for TokenBatchIteratorOptions {
    fn default() -> Self {
        Self {
            batch_size: 32,
            batch_seq_len: 2048,
            min_buffer: 1024,
        }
    }
}
