//! # Thread Regex Pool

use ahash::AHashMap;
use fancy_regex::Regex;
use std::sync::{Arc, RwLock};
use std::thread::ThreadId;

/// Interior-Mutable Thread-Local Regex Pool
#[derive(Clone)]
pub struct RegexPool {
    regex: Regex,

    pool: Arc<RwLock<AHashMap<ThreadId, Arc<Regex>>>>,
}

impl RegexPool {
    /// Create a new `RegexPool`
    pub fn new(regex: Regex) -> Self {
        Self {
            regex,
            pool: Arc::new(RwLock::new(AHashMap::new())),
        }
    }

    /// Get a Regex from the pool for the current thread.
    pub fn get(&self) -> Arc<Regex> {
        let thread_id = std::thread::current().id();

        if let Some(regex) = self.pool.read().unwrap().get(&thread_id) {
            return regex.clone();
        }

        let mut write_lock = self.pool.write().unwrap();
        write_lock
            .entry(thread_id)
            .or_insert_with(|| Arc::new(self.regex.clone()))
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regex_pool() {
        let regex = Regex::new(r"foo").unwrap();
        let pool = RegexPool::new(regex);

        let r0 = pool.get();
        assert_eq!(r0.as_str(), r"foo");

        assert!(Arc::ptr_eq(&r0, &pool.get()));
    }
}
