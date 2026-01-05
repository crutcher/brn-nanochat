//! Pair Count / Word Indexing

use crate::types::{CountType, Pair, TokenType};
use crate::vocab::training::word::Word;
use ahash::{AHashMap, AHashSet};

/// Options for building a [`PairIndex`].
#[derive(Debug, Clone, Copy)]
pub struct PairIndexOptions {
    /// Whether to use parallel processing for indexing.
    ///
    /// Requires the `rayon` feature to be enabled.
    pub parallel: bool,
}

impl Default for PairIndexOptions {
    fn default() -> Self {
        Self {
            parallel: crate::DEFAULT_PARALLEL,
        }
    }
}

impl PairIndexOptions {
    /// Sets the parallel processing flag.
    pub fn with_parallel(
        self,
        parallel: bool,
    ) -> Self {
        Self { parallel }
    }
}

/// An index of [`Pair`]s over an index set of ``(word, count)``.
#[derive(Debug)]
pub struct PairIndex<T: TokenType, C: CountType> {
    /// A map from [`Pair`] to its occurrence count.
    ///
    /// ``sum(words[i].non_overlapping_count(pair) * word_counts[i]) for all i``
    pub pair_counts: AHashMap<Pair<T>, C>,

    /// A map from [`Pair`] to indices over ``words``.
    pub pair_to_word_index: AHashMap<Pair<T>, AHashSet<usize>>,
}

impl<T: TokenType, C: CountType> PairIndex<T, C> {
    /// Build a [`PairIndex`] from a slice of [`Word`]s, using a count table.
    ///
    /// # Arguments
    /// * `words` - the slice of words; Words are assumed to be unique.
    /// * `word_counts` - `word_counts[i]` is the count of `words[i]`.
    /// * `options` - options for building the index.
    #[tracing::instrument(skip(words, word_counts))]
    pub fn index_unique_word_counts_table(
        words: &[Word<T>],
        word_counts: &[C],
        options: PairIndexOptions,
    ) -> Self {
        if options.parallel {
            #[cfg(not(feature = "rayon"))]
            panic!("Parallel processing requires the `rayon` feature to be enabled.");

            #[cfg(feature = "rayon")]
            Self::index_unique_word_counts_table_rayon(words, word_counts, options)
        } else {
            Self::index_unique_word_counts_table_serial(words, word_counts, options)
        }
    }

    #[tracing::instrument(skip(pair_counts, pair_to_word_index, index, w, word_count))]
    fn observe_word(
        pair_counts: &mut AHashMap<Pair<T>, C>,
        pair_to_word_index: &mut AHashMap<Pair<T>, AHashSet<usize>>,
        index: usize,
        w: &Word<T>,
        word_count: C,
    ) {
        if word_count != C::zero() && w.len() >= 2 {
            for p in w.pairs() {
                *pair_counts.entry(p).or_default() += word_count;
                pair_to_word_index.entry(p).or_default().insert(index);
            }
        }
    }

    /// Build a [`PairIndex`] from a slice of [`Word`]s, using a count table.
    ///
    /// This is a serial implementation that does not use parallelism;
    /// this ignores the `options.parallel` flag.
    ///
    /// # Arguments
    /// * `words` - the slice of words.
    /// * `words` - the slice of words; Words are assumed to be unique.
    /// * `options` - options for building the index.
    pub fn index_unique_word_counts_table_serial(
        words: &[Word<T>],
        word_counts: &[C],
        _options: PairIndexOptions,
    ) -> Self {
        let mut pair_counts: AHashMap<Pair<T>, C> = Default::default();
        let mut pair_to_word_index: AHashMap<Pair<T>, AHashSet<usize>> = Default::default();

        for (word_index, word) in words.iter().enumerate() {
            Self::observe_word(
                &mut pair_counts,
                &mut pair_to_word_index,
                word_index,
                word,
                word_counts[word_index],
            );
        }

        Self {
            pair_counts,
            pair_to_word_index,
        }
    }

    /// Build a [`PairIndex`] from a slice of [`Word`]s, using a count table.
    ///
    /// This is a `rayon` implementation that uses parallelism;
    /// this ignores the `options.parallel` flag.
    ///
    /// # Arguments
    /// * `words` - the slice of words; Words are assumed to be unique.
    /// * `word_counts` - `word_counts[i]` is the duplication count of `words[i]`.
    /// * `options` - options for building the index.
    #[cfg(feature = "rayon")]
    pub fn index_unique_word_counts_table_rayon(
        words: &[Word<T>],
        word_counts: &[C],
        _options: PairIndexOptions,
    ) -> Self {
        use rayon::prelude::*;

        let (pair_counts, pair_to_word_index) = words
            .par_iter()
            .enumerate()
            .map(|(word_index, word)| {
                let mut local_pc: AHashMap<Pair<T>, C> = Default::default();
                let mut local_wtu: AHashMap<Pair<T>, AHashSet<usize>> = Default::default();
                Self::observe_word(
                    &mut local_pc,
                    &mut local_wtu,
                    word_index,
                    word,
                    word_counts[word_index],
                );
                (local_pc, local_wtu)
            })
            .reduce(Default::default, |(mut acc_pc, mut acc_wtu), (pc, wtu)| {
                for (k, v) in pc {
                    *acc_pc.entry(k).or_default() += v;
                }
                for (k, s) in wtu {
                    acc_wtu.entry(k).or_default().extend(s);
                }
                (acc_pc, acc_wtu)
            });

        Self {
            pair_counts,
            pair_to_word_index,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MergeMap;
    use crate::vocab::training::word::Word;

    #[test]
    fn test_pair_index_serial() {
        test_pair_index(false);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_pair_index_parallel() {
        test_pair_index(true);
    }

    fn test_pair_index(parallel: bool) {
        type T = u8;

        let words = vec![
            Word::from_string("hello"),
            Word::from_string("world"),
            Word::from_string("help"),
        ];

        let word_counts = vec![1, 2, 3];

        let options = PairIndexOptions::default().with_parallel(parallel);

        let index = PairIndex::index_unique_word_counts_table(&words, &word_counts, options);

        let PairIndex {
            pair_counts,
            pair_to_word_index,
        } = index;

        assert_eq!(
            pair_counts,
            [
                (('e', 'l'), 4),
                (('h', 'e'), 4),
                (('l', 'd'), 2),
                (('l', 'l'), 1),
                (('l', 'o'), 1),
                (('l', 'p'), 3),
                (('o', 'r'), 2),
                (('r', 'l'), 2),
                (('w', 'o'), 2),
            ]
            .into_iter()
            .map(|((a, b), c)| ((a as u8, b as u8), c))
            .collect::<MergeMap<T>>()
        );

        assert_eq!(
            pair_to_word_index,
            [
                (('e', 'l'), vec![0, 2]),
                (('h', 'e'), vec![0, 2]),
                (('l', 'd'), vec![1]),
                (('l', 'l'), vec![0]),
                (('l', 'o'), vec![0]),
                (('l', 'p'), vec![2]),
                (('o', 'r'), vec![1]),
                (('r', 'l'), vec![1]),
                (('w', 'o'), vec![1]),
            ]
            .into_iter()
            .map(|((a, b), s)| ((a as u8, b as u8), AHashSet::from_iter(s)))
            .collect::<AHashMap<Pair<T>, AHashSet<usize>>>()
        );
    }
}
