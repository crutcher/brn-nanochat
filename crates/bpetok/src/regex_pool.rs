//! # Thread Regex Pool

use ahash::AHashMap;
use fancy_regex::Regex;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, RwLock};

/// Interior-Mutable Thread-Local Regex Pool
#[derive(Clone)]
pub struct RegexPool {
    regex: Regex,

    pool_size: usize,
    pool: Arc<RwLock<AHashMap<usize, Arc<Regex>>>>,
}

fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

impl RegexPool {
    /// Create a new `RegexPool`
    pub fn new(
        regex: Regex,
        pool_size: usize,
    ) -> Self {
        Self {
            regex,
            pool_size,
            pool: Arc::new(RwLock::new(AHashMap::new())),
        }
    }

    /// Get a Regex from the pool for the current thread.
    pub fn get(&self) -> Arc<Regex> {
        let thread_id = std::thread::current().id();
        let slot = calculate_hash(&thread_id) as usize / self.pool_size;

        if let Some(regex) = self.pool.read().unwrap().get(&slot) {
            return regex.clone();
        }

        let mut write_lock = self.pool.write().unwrap();
        write_lock
            .entry(slot)
            .or_insert(Arc::new(self.regex.clone()))
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regex_pool() {
        let regex = Regex::new(r"foo").unwrap();
        let pool = RegexPool::new(regex, 10);

        let r0 = pool.get();
        assert_eq!(r0.as_str(), r"foo");

        assert!(Arc::ptr_eq(&r0, &pool.get()));
    }
}
