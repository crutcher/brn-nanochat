use std::sync::{
    Arc,
    atomic::AtomicUsize,
};

/// An iterator that counts the number of items it has yielded.
#[derive(Debug)]
pub struct CountingIter<I>
where
    I: Iterator,
{
    inner: I,
    counter: Arc<AtomicUsize>,
}

impl<I> CountingIter<I>
where
    I: Iterator,
{
    /// Create a new `CountingIter` from an inner iterator.
    pub fn new(inner: I) -> Self {
        Self {
            inner,
            counter: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn counter(&self) -> Arc<AtomicUsize> {
        self.counter.clone()
    }
}

impl<I> Iterator for CountingIter<I>
where
    I: Iterator,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counting_iter() {
        let mut iter = CountingIter::new(vec![1, 2, 3].into_iter());

        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.counter().load(std::sync::atomic::Ordering::Relaxed), 2);
    }
}
