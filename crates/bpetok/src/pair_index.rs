//! Pair Count / Word Indexing

use crate::types::{CountType, Pair, TokenType};
use crate::word::Word;
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
    use crate::word::Word;

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
        let words = vec![
            Word::from_tokens(['h' as u8, 'e' as u8, 'l' as u8, 'l' as u8, 'o' as u8]),
            Word::from_tokens(['w' as u8, 'o' as u8, 'r' as u8, 'l' as u8, 'd' as u8]),
            Word::from_tokens(['h' as u8, 'e' as u8, 'l' as u8, 'p' as u8]),
        ];

        let word_counts = vec![1, 2, 3];

        let options = PairIndexOptions::default().with_parallel(parallel);

        let index = PairIndex::index_unique_word_counts_table(&words, &word_counts, options);

        let PairIndex {
            pair_counts,
            pair_to_word_index,
        } = index;

        let mut pair_counts: Vec<_> = pair_counts.into_iter().collect();
        pair_counts.sort();
        assert_eq!(
            pair_counts,
            vec![
                (('e' as u8, 'l' as u8), 4),
                (('h' as u8, 'e' as u8), 4),
                (('l' as u8, 'd' as u8), 2),
                (('l' as u8, 'l' as u8), 1),
                (('l' as u8, 'o' as u8), 1),
                (('l' as u8, 'p' as u8), 3),
                (('o' as u8, 'r' as u8), 2),
                (('r' as u8, 'l' as u8), 2),
                (('w' as u8, 'o' as u8), 2),
            ]
        );

        let mut pair_to_word_index: Vec<_> = pair_to_word_index
            .into_iter()
            .map(|(p, wi)| {
                let mut wi = wi.into_iter().collect::<Vec<_>>();
                wi.sort();
                (p, wi)
            })
            .collect();
        pair_to_word_index.sort_by_key(|(p, _)| *p);
        assert_eq!(
            pair_to_word_index,
            vec![
                (('e' as u8, 'l' as u8), vec![0, 2]),
                (('h' as u8, 'e' as u8), vec![0, 2]),
                (('l' as u8, 'd' as u8), vec![1]),
                (('l' as u8, 'l' as u8), vec![0]),
                (('l' as u8, 'o' as u8), vec![0]),
                (('l' as u8, 'p' as u8), vec![2]),
                (('o' as u8, 'r' as u8), vec![1]),
                (('r' as u8, 'l' as u8), vec![1]),
                (('w' as u8, 'o' as u8), vec![1]),
            ]
        );
    }
}
