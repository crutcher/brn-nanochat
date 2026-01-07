//! # Thread Regex Pool

use crate::util::regex::regex_wrapper::RegexWrapper;
use ahash::AHashMap;
use std::num::NonZero;
use std::sync::{Arc, RwLock};
use std::thread::ThreadId;

fn unsafe_threadid_to_u64(thread_id: &ThreadId) -> u64 {
    unsafe { std::mem::transmute(thread_id) }
}

/// Interior-Mutable Thread-Local Regex Pool
#[derive(Clone)]
pub struct RegexWrapperPool {
    regex: Arc<RegexWrapper>,

    max_pool: u64,
    pool: Arc<RwLock<AHashMap<u64, Arc<RegexWrapper>>>>,
}

impl From<Arc<RegexWrapper>> for RegexWrapperPool {
    fn from(regex: Arc<RegexWrapper>) -> Self {
        Self::new(regex)
    }
}

impl RegexWrapperPool {
    /// Create a new `RegexPool`
    pub fn new(regex: Arc<RegexWrapper>) -> Self {
        let max_pool = std::thread::available_parallelism()
            .unwrap_or(NonZero::new(128).unwrap())
            .get() as u64;

        Self {
            regex,
            max_pool,
            pool: Arc::new(RwLock::new(AHashMap::new())),
        }
    }

    /// Clear the regex pool.
    pub fn clear(&self) {
        self.pool.write().unwrap().clear();
    }

    /// Get a Regex from the pool for the current thread.
    pub fn get(&self) -> Arc<RegexWrapper> {
        let thread_id = std::thread::current().id();
        let slot = unsafe_threadid_to_u64(&thread_id) % self.max_pool;

        if let Some(regex) = self.pool.read().unwrap().get(&slot) {
            return regex.clone();
        }

        let mut writer = self.pool.write().unwrap();
        let re = Arc::new((*self.regex).clone());
        writer.insert(slot, re.clone());
        re
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::regex::regex_wrapper::RegexPatternLabel;

    #[test]
    fn test_regex_pool() {
        let pattern: RegexPatternLabel = r"foo".into();
        let regex = pattern.compile().unwrap().into();

        let pool = RegexWrapperPool::new(regex);

        let r0 = pool.get();
        assert_eq!(r0.as_str(), r"foo");

        assert!(Arc::ptr_eq(&r0, &pool.get()));
    }
}
