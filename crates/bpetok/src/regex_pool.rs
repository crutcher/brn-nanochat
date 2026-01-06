//! # Thread Regex Pool
#![allow(unused)]

use ahash::AHashMap;
use fancy_regex::Regex;
use std::hash::{DefaultHasher, Hasher};
use std::num::NonZero;
use std::ptr::hash;
use std::sync::{Arc, RwLock};

/// Interior-Mutable Thread-Local Regex Pool
#[derive(Clone)]
pub struct RegexPool {
    regex: Arc<Regex>,

    max_pool: u64,
    pool: Arc<RwLock<AHashMap<u64, Arc<Regex>>>>,
}

impl RegexPool {
    /// Create a new `RegexPool`
    pub fn new(regex: Regex) -> Self {
        let max_pool = std::thread::available_parallelism()
            .unwrap_or(NonZero::new(128).unwrap())
            .get() as u64;

        Self {
            regex: Arc::new(regex),
            max_pool,
            pool: Arc::new(RwLock::new(AHashMap::new())),
        }
    }

    /// Get a Regex from the pool for the current thread.
    pub fn get(&self) -> Arc<Regex> {
        self.regex.clone()
        /*
        let thread_id = std::thread::current().id();

        let mut s = DefaultHasher::new();
        hash(&thread_id, &mut s);
        let slot = s.finish();

        let slot = slot % self.max_pool;

        if let Some(regex) = self.pool.read().unwrap().get(&slot) {
            return regex.clone();
        }

        let mut writer = self.pool.write().unwrap();
        let re = Arc::new((*self.regex).clone());
        writer.insert(slot, re.clone());
        re

         */
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
