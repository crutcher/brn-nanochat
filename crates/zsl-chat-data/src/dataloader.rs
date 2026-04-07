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
        IterWatcher,
        ShuffleIter,
    },
    tokens::{
        DenseTokenBlocksOptions,
        tokenize_text_batches,
    },
};

pub struct EpochStats {
    file_counter: Arc<AtomicUsize>,
    byte_counter: Arc<AtomicUsize>,
    token_counter: Arc<AtomicUsize>,
    items_total: usize,
}

impl EpochStats {
    pub fn new(
        file_counter: Arc<AtomicUsize>,
        byte_counter: Arc<AtomicUsize>,
        token_counter: Arc<AtomicUsize>,
        items_total: usize,
    ) -> Self {
        Self {
            file_counter,
            byte_counter,
            token_counter,
            items_total,
        }
    }

    pub fn file_count(&self) -> usize {
        self.file_counter.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn byte_count(&self) -> usize {
        self.byte_counter.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn token_count(&self) -> usize {
        self.token_counter
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn items_total(&self) -> usize {
        self.items_total
    }

    pub fn progress(&self) -> Progress {
        Progress {
            items_processed: self.file_count(),
            items_total: self.items_total(),
        }
    }
}

pub struct ChatDataLoaderIterator {
    stats: Arc<EpochStats>,
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

        let file_counter = Arc::new(AtomicUsize::new(0));
        let file_counter_handle = file_counter.clone();
        let shard_counter = IterWatcher::new(
            shard_paths.into_iter(),
            Box::new(move |_| {
                file_counter_handle.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }),
        );

        // Iterator<ArrowResult<RecordBatch>>
        let parquet_batches = read_parquet_shards(shard_counter);

        // Iterator<ArrowResult<Vec<String>>>
        let byte_counter = Arc::new(AtomicUsize::new(0));
        let byte_counter_handle = byte_counter.clone();
        let sample_batches = IterWatcher::new(
            select_text_column(text_column, parquet_batches),
            Box::new(move |result| {
                if let Ok(batch) = &result {
                    let bytes = batch.iter().map(|s| s.as_str().len()).sum::<usize>();
                    byte_counter_handle.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
                }
            }),
        );

        // Iterator<ArrowResult<Vec<Vec<u32>>>>
        let token_batches = tokenize_text_batches(tokenizer, sample_batches);

        // Iterator<ArrowResult<Vec<Vec<u32>>>> (batch_size x batch_seq_len)
        let token_counter = Arc::new(AtomicUsize::new(0));
        let token_counter_handle = token_counter.clone();
        let dense_blocks = IterWatcher::new(
            block_options.build_dense_blocks(token_batches),
            Box::new(move |result| {
                if let Ok(batch) = &result {
                    let tokens = batch.iter().map(|ts| ts.len()).sum::<usize>();
                    token_counter_handle.fetch_add(tokens, std::sync::atomic::Ordering::Relaxed);
                }
            }),
        );

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
            stats: Arc::new(EpochStats {
                file_counter,
                token_counter,
                byte_counter,
                items_total,
            }),
            inner,
        }
    }

    pub fn stats(&self) -> &Arc<EpochStats> {
        &self.stats
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
        self.stats.progress()
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

    /// Starts a new epoch.
    pub fn start_epoch(&self) -> ChatDataLoaderIterator {
        let mut shard_paths = self.shard_paths.clone();
        if let Some(mutex) = &self.rng {
            let mut rng = mutex.lock().unwrap();
            shard_paths.shuffle(&mut *rng);
        }

        let shuffle_buffer_fill_rate = 2;
        let shuffle_buffer_size = if self.rng.is_none() { 0 } else { 128 };

        ChatDataLoaderIterator::new(
            self.tokenizer.clone(),
            shard_paths,
            self.block_options.clone(),
            shuffle_buffer_fill_rate,
            shuffle_buffer_size,
            "text",
        )
    }
}

impl<B: Backend> DataLoader<B, Result<Vec<Vec<u32>>, ArrowError>> for ChatDataLoader<B>
where
    B: Backend,
{
    fn iter(&self) -> Box<dyn DataLoaderIterator<Result<Vec<Vec<u32>>, ArrowError>>> {
        Box::new(self.start_epoch())
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
