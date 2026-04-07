use rand::RngExt;

/// An iterator that shuffles the items it iterates over.
pub struct ShuffleIter<I, T>
where
    I: Iterator<Item = T>,
{
    inner: I,
    done: bool,
    buffer: Vec<T>,
    fill_rate: usize,
    buffer_size: usize,
    rng: Box<dyn rand::Rng + Send>,
}

impl<I, T> ShuffleIter<I, T>
where
    I: Iterator<Item = T>,
{
    /// Creates a new `ShuffleIter`.
    ///
    /// Using `fill_rate === buffer_size` will produce the common semantics of
    /// af streaming shuffle. Using a smaller `fill_rate` can reduce latency
    /// at the begining of iteration, at the cost of shuffle bias.
    ///
    /// ## Arguments
    /// * `inner` - The iterator to shuffle.
    /// * `fill_rate` - The upper limit on calls to `inner.next()` per step.
    /// * `buffer_size` - The size of the shuffle buffer.
    /// * `rng` - The random number generator to use.
    pub fn new(
        inner: I,
        fill_rate: usize,
        buffer_size: usize,
        rng: Box<dyn rand::Rng + Send>,
    ) -> Self {
        assert!(fill_rate > 0);
        assert!(fill_rate <= buffer_size);
        assert!(buffer_size > 1);
        Self {
            inner,
            done: false,
            buffer: Vec::with_capacity(buffer_size),
            fill_rate,
            buffer_size,
            rng,
        }
    }
}

impl<I, T> Iterator for ShuffleIter<I, T>
where
    I: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        for _ in 0..self.fill_rate {
            if !self.done && self.buffer.len() < self.buffer_size {
                if let Some(item) = self.inner.next() {
                    self.buffer.push(item);
                    continue;
                } else {
                    self.done = true;
                }
            }
            break;
        }

        if self.buffer.is_empty() {
            return None;
        }

        let idx = self.rng.random_range(0..self.buffer.len());
        let val = self.buffer.swap_remove(idx);

        if !self.done
            && let Some(new_value) = self.inner.next()
        {
            self.buffer.push(new_value);
        }

        Some(val)
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn test_shuffle_iter() {
        let items = (0..10).collect::<Vec<u32>>();

        let shuffle_buffer_fill_rate = 2;
        let shuffle_buffer_size = 5;

        let iter = ShuffleIter::new(
            items.clone().into_iter(),
            shuffle_buffer_fill_rate,
            shuffle_buffer_size,
            Box::new(rand::rngs::StdRng::seed_from_u64(0)),
        );

        let mut results = iter.collect::<Vec<u32>>();

        results.sort_unstable();

        assert_eq!(results, items);
    }
}
