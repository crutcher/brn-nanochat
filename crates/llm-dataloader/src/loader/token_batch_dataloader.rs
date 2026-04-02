use std::sync::Arc;

use burn::{
    data::dataloader::{
        DataLoader,
        DataLoaderIterator,
    },
    prelude::Backend,
};
use wordchipper::TokenType;

use crate::loader::{
    TokenBatch,
    TokenBatchIteratorFactory,
};

pub struct TokenBatchDataLoader<T: TokenType, B: Backend> {
    loader: TokenBatchIteratorFactory<T>,
    device: B::Device,
    shuffle: bool,
}

impl<T: TokenType, B: Backend> TokenBatchDataLoader<T, B> {
    pub fn new(
        loader: TokenBatchIteratorFactory<T>,
        device: B::Device,
        shuffle: bool,
    ) -> Self {
        Self {
            loader,
            device,
            shuffle,
        }
    }
}

impl<T: TokenType, B: Backend> DataLoader<B, TokenBatch<T>> for TokenBatchDataLoader<T, B> {
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<TokenBatch<T>> + 'a> {
        Box::new(self.loader.iter(self.shuffle))
    }

    fn num_items(&self) -> usize {
        self.loader.num_items()
    }

    fn to_device(
        &self,
        device: &B::Device,
    ) -> Arc<dyn DataLoader<B, TokenBatch<T>>> {
        Arc::new(Self::new(self.loader.clone(), device.clone(), self.shuffle))
    }

    fn slice(
        &self,
        start: usize,
        end: usize,
    ) -> Arc<dyn DataLoader<B, TokenBatch<T>>> {
        Arc::new(Self::new(
            self.loader.slice(start, end),
            self.device.clone(),
            self.shuffle,
        ))
    }
}
