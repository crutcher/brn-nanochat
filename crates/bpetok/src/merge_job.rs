use crate::Pair;
use crate::token_types::TokenType;
use ahash::AHashSet;
use core::cmp::Ordering;

/// Info about a [`Pair`] that could be merged.
#[derive(Debug, Eq)]
pub struct MergeJob<T: TokenType> {
    /// The pair to merge.
    pub pair: Pair<T>,

    /// The number of instances of this pair in the corpus.
    pub count: usize,

    /// Word indices that may contain this pair.
    pub word_indices: AHashSet<usize>,
}

impl<T: TokenType> MergeJob<T> {
    /// The job key.
    ///
    /// Max-heap by count; tie-break to ascending pair order (deterministic)
    pub fn heap_key(&self) -> (usize, Pair<T>) {
        (self.count, self.pair)
    }
}

impl<T: TokenType> PartialEq for MergeJob<T> {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.heap_key() == other.heap_key()
    }
}

impl<T: TokenType> PartialOrd for MergeJob<T> {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: TokenType> Ord for MergeJob<T> {
    fn cmp(
        &self,
        other: &Self,
    ) -> Ordering {
        self.heap_key().cmp(&other.heap_key())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_job_heap_key() {
        let job1 = MergeJob {
            pair: (1, 2),
            count: 2,
            word_indices: AHashSet::new(),
        };

        let job2 = MergeJob {
            pair: (2, 1),
            count: 1,
            word_indices: AHashSet::new(),
        };
        let job3 = MergeJob {
            pair: (2, 2),
            count: 1,
            word_indices: AHashSet::new(),
        };

        assert_eq!(job1.heap_key(), (2, (1, 2)));
        assert_eq!(job2.heap_key(), (1, (2, 1)));

        assert_eq!(job1.heap_key().cmp(&job1.heap_key()), Ordering::Equal);
        assert_eq!(
            job1.heap_key().partial_cmp(&job1.heap_key()),
            Some(Ordering::Equal)
        );

        assert_eq!(job2.heap_key().cmp(&job2.heap_key()), Ordering::Equal);

        assert_eq!(job1.heap_key().cmp(&job2.heap_key()), Ordering::Greater);
        assert_eq!(job2.heap_key().cmp(&job1.heap_key()), Ordering::Less);

        assert_eq!(job3.heap_key().cmp(&job2.heap_key()), Ordering::Greater);
        assert_eq!(job2.heap_key().cmp(&job3.heap_key()), Ordering::Less);
    }
}
