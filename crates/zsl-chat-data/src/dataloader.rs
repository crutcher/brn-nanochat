use std::{
    path::PathBuf,
    sync::{
        Arc,
        Mutex,
        atomic::AtomicUsize,
    },
};

use arrow::error::ArrowError;
use burn::{
    data::dataloader::{
        DataLoader,
        DataLoaderIterator,
        Progress,
    },
    prelude::Backend,
};
use rand::{
    SeedableRng,
    rngs::StdRng,
    seq::SliceRandom,
};
use wordchipper::Tokenizer;

use crate::{
    arrow::{
        read_parquet_shards,
        select_text_column,
    },
    iterators::{
        CountingIter,
        ShuffleIter,
    },
    tokens::{
        DenseTokenBlocksOptions,
        tokenize_text_batches,
    },
};

pub struct ChatDataLoaderIterator {
    file_counter: Arc<AtomicUsize>,
    items_total: usize,
    inner: Box<dyn Iterator<Item = Result<Vec<Vec<u32>>, ArrowError>>>,
}

impl ChatDataLoaderIterator {
    pub fn new(
        tokenizer: Arc<Tokenizer<u32>>,
        shard_paths: Vec<PathBuf>,
        block_options: DenseTokenBlocksOptions,
        shuffle_buffer_fill_rate: usize,
        shuffle_buffer_size: usize,
        text_column: &str,
    ) -> Self {
        let items_total = shard_paths.len();

        let shard_counter = CountingIter::new(shard_paths.into_iter());
        let file_counter = shard_counter.counter();

        // Iterator<ArrowResult<RecordBatch>>
        let parquet_batches = read_parquet_shards(shard_counter);

        // Iterator<ArrowResult<Vec<String>>>
        let sample_batches = select_text_column(text_column, parquet_batches);

        // Iterator<ArrowResult<Vec<Vec<u32>>>>
        let token_batches = tokenize_text_batches(tokenizer, sample_batches);

        // Iterator<ArrowResult<Vec<Vec<u32>>>> (batch_size x batch_seq_len)
        let dense_blocks = block_options.build_dense_blocks(token_batches);

        // TODO: pass in rng
        let inner: Box<dyn Iterator<Item = Result<Vec<Vec<u32>>, ArrowError>>> =
            if shuffle_buffer_size == 0 {
                Box::new(dense_blocks)
            } else {
                Box::new(ShuffleIter::new(
                    dense_blocks,
                    shuffle_buffer_fill_rate,
                    shuffle_buffer_size,
                    Box::new(StdRng::seed_from_u64(0)),
                ))
            };

        Self {
            file_counter,
            items_total,
            inner,
        }
    }
}

impl Iterator for ChatDataLoaderIterator {
    type Item = Result<Vec<Vec<u32>>, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl DataLoaderIterator<Result<Vec<Vec<u32>>, ArrowError>> for ChatDataLoaderIterator {
    fn progress(&self) -> Progress {
        Progress {
            items_processed: self.file_counter.load(std::sync::atomic::Ordering::Relaxed),
            items_total: self.items_total,
        }
    }
}

pub struct ChatDataLoader<B: Backend> {
    shard_paths: Vec<PathBuf>,
    rng: Option<Arc<Mutex<dyn rand::Rng + Send>>>,
    device: B::Device,
    tokenizer: Arc<Tokenizer<u32>>,
    block_options: DenseTokenBlocksOptions,
}

impl<B: Backend> ChatDataLoader<B> {
    pub fn new(
        files: Vec<PathBuf>,
        rng: Option<Arc<Mutex<dyn rand::Rng + Send>>>,
        device: &B::Device,
        tokenizer: Arc<Tokenizer<u32>>,
        block_options: DenseTokenBlocksOptions,
    ) -> Self {
        Self {
            shard_paths: files,
            rng,
            device: device.clone(),
            tokenizer,
            block_options,
        }
    }
}

impl<B: Backend> DataLoader<B, Result<Vec<Vec<u32>>, ArrowError>> for ChatDataLoader<B>
where
    B: Backend,
{
    fn iter(&self) -> Box<dyn DataLoaderIterator<Result<Vec<Vec<u32>>, ArrowError>>> {
        let mut shard_paths = self.shard_paths.clone();
        if let Some(mutex) = &self.rng {
            let mut rng = mutex.lock().unwrap();
            shard_paths.shuffle(&mut *rng);
        }

        let shuffle_buffer_fill_rate = 2;
        let shuffle_buffer_size = if self.rng.is_none() { 0 } else { 128 };

        Box::new(ChatDataLoaderIterator::new(
            self.tokenizer.clone(),
            shard_paths,
            self.block_options.clone(),
            shuffle_buffer_fill_rate,
            shuffle_buffer_size,
            "text",
        ))
    }

    fn num_items(&self) -> usize {
        self.shard_paths.len()
    }

    fn to_device(
        &self,
        device: &B::Device,
    ) -> Arc<dyn DataLoader<B, Result<Vec<Vec<u32>>, ArrowError>>> {
        Arc::new(Self {
            shard_paths: self.shard_paths.clone(),
            rng: self.rng.clone(),
            device: device.clone(),
            tokenizer: self.tokenizer.clone(),
            block_options: self.block_options.clone(),
        })
    }

    fn slice(
        &self,
        start: usize,
        end: usize,
    ) -> Arc<dyn DataLoader<B, Result<Vec<Vec<u32>>, ArrowError>>> {
        Arc::new(Self {
            shard_paths: self.shard_paths[start..end].to_vec(),
            rng: self.rng.clone(),
            device: self.device.clone(),
            tokenizer: self.tokenizer.clone(),
            block_options: self.block_options.clone(),
        })
    }
}
