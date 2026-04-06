use rand::RngExt;

pub struct ShuffleIter<I, T>
where
    I: Iterator<Item = T>,
{
    inner: I,
    done: bool,
    buffer: Vec<T>,
    capacity: usize,
    rng: Box<dyn rand::Rng + Send>,
}

impl<I, T> ShuffleIter<I, T>
where
    I: Iterator<Item = T>,
{
    pub fn new(
        inner: I,
        capacity: usize,
        rng: Box<dyn rand::Rng + Send>,
    ) -> Self {
        Self {
            inner,
            done: false,
            buffer: Vec::with_capacity(capacity),
            capacity,
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
        while !self.done && self.buffer.len() < self.capacity {
            if let Some(item) = self.inner.next() {
                self.buffer.push(item);
            } else {
                self.done = true;
            }
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
