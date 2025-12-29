//! Pair Count / Word Indexing
use crate::token_types::Token;
use crate::{Pair, Word};
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

/// An index of [`Pair`]s over an index set of ``(word, count)``.
#[derive(Debug)]
pub struct PairIndex<T: Token> {
    /// A map from [`Pair`] to its occurrence count.
    ///
    /// ``sum(words[i].non_overlapping_count(pair) * word_counts[i]) for all i``
    pub pair_counts: AHashMap<Pair<T>, usize>,

    /// A map from [`Pair`] to indices over ``words``.
    pub pair_to_word_index: AHashMap<Pair<T>, AHashSet<usize>>,
}

impl<T: Token> PairIndex<T> {
    /// Build a [`PairIndex`] from a slice of [`Word`]s, using a count table.
    ///
    /// # Arguments
    /// * `words` - the slice of words; Words are assumed to be unique.
    /// * `word_counts` - `word_counts[i]` is the count of `words[i]`.
    /// * `options` - options for building the index.
    pub fn index_unique_word_counts_table(
        words: &[Word<T>],
        word_counts: &[usize],
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
        word_counts: &[usize],
        _options: PairIndexOptions,
    ) -> Self {
        let mut pair_counts: AHashMap<Pair<T>, usize> = Default::default();
        let mut pair_to_word_index: AHashMap<Pair<T>, AHashSet<usize>> = Default::default();

        for (i, w) in words.iter().enumerate() {
            let wc = word_counts[i];
            if wc != 0 && w.len() >= 2 {
                for p in w.pairs() {
                    *pair_counts.entry(p).or_default() += wc;
                    pair_to_word_index.entry(p).or_default().insert(i);
                }
            }
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
        word_counts: &[usize],
        _options: PairIndexOptions,
    ) -> Self {
        use rayon::prelude::*;

        let (pair_counts, pair_to_word_index) = words
            .par_iter()
            .enumerate()
            .map(|(i, w)| {
                let mut local_pc: AHashMap<Pair<T>, usize> = AHashMap::new();
                let mut local_wtu: AHashMap<Pair<T>, AHashSet<usize>> = AHashMap::new();
                let wc = word_counts[i];
                if wc != 0 && w.len() >= 2 {
                    for p in w.pairs() {
                        *local_pc.entry(p).or_default() += word_counts[i];
                        local_wtu.entry(p).or_default().insert(i);
                    }
                }
                (local_pc, local_wtu)
            })
            .reduce(
                || (AHashMap::new(), AHashMap::new()),
                |(mut acc_pc, mut acc_wtu), (pc, wtu)| {
                    for (k, v) in pc {
                        *acc_pc.entry(k).or_default() += v;
                    }
                    for (k, s) in wtu {
                        acc_wtu.entry(k).or_default().extend(s);
                    }
                    (acc_pc, acc_wtu)
                },
            );

        Self {
            pair_counts,
            pair_to_word_index,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pair_index() {
        let words = vec![
            Word::from_tokens(['h' as u8, 'e' as u8, 'l' as u8, 'l' as u8, 'o' as u8]),
            Word::from_tokens(['w' as u8, 'o' as u8, 'r' as u8, 'l' as u8, 'd' as u8]),
            Word::from_tokens(['h' as u8, 'e' as u8, 'l' as u8, 'p' as u8]),
        ];

        let word_counts = vec![1, 2, 3];

        for parallel in [true, false] {
            let options = PairIndexOptions { parallel };

            let index =
                PairIndex::index_unique_word_counts_table_rayon(&words, &word_counts, options);

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
}
