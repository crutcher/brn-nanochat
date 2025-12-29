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

/// Compute the word count table for a sequence of words.
pub fn word_count_table<T: Token>(words: &[Word<T>]) -> Vec<usize> {
    let mut word_counts: AHashMap<&Word<T>, usize> = Default::default();

    words.iter().for_each(|w| {
        *word_counts.entry(w).or_default() += 1;
    });

    words.iter().map(|w| word_counts[w]).collect()
}

/// An index of [`Pair`]s over a sequence of [`Word`]s.
#[derive(Debug)]
pub struct PairIndex<T: Token> {
    /// A map from pair to its total count across all words.
    pub pair_counts: AHashMap<Pair<T>, usize>,

    /// A map from pair to the set of word indices that contain it.
    pub pair_to_word_index: AHashMap<Pair<T>, AHashSet<usize>>,
}

impl<T: Token> PairIndex<T> {
    /// Create a [`PairIndex`] from a sequence of [`Word`]s.
    ///
    /// # Arguments
    /// * `words` - the slice of words.
    /// * `options` - options for building the index.
    pub fn for_words(
        words: &[Word<T>],
        options: PairIndexOptions,
    ) -> Self {
        let mut word_counts: AHashMap<&Word<T>, usize> = Default::default();

        words.iter().for_each(|w| {
            *word_counts.entry(w).or_default() += 1;
        });

        Self::for_words_with_count_fn(words, options, |w| word_counts[w])
    }

    /// Create a [`PairIndex`] from a sequence of [`Word`]s, using a count function.
    ///
    /// # Arguments
    /// * `words` - the slice of words.
    /// * `options` - options for building the index.
    /// * `count_fn` - `count_fn(&word)` is the duplication count of `&word`.
    pub fn for_words_with_count_fn<F>(
        words: &[Word<T>],
        options: PairIndexOptions,
        count_fn: F,
    ) -> Self
    where
        F: Fn(&Word<T>) -> usize,
    {
        Self::for_words_with_count_table(
            words,
            options,
            &words.iter().map(count_fn).collect::<Vec<_>>(),
        )
    }

    /// Build a [`PairIndex`] from a slice of [`Word`]s, using a count table.
    ///
    /// # Arguments
    /// * `words` - the slice of words.
    /// * `options` - options for building the index.
    /// * `word_counts` - `word_counts[i]` is the duplication count of `words[i]`.
    pub fn for_words_with_count_table(
        words: &[Word<T>],
        options: PairIndexOptions,
        word_counts: &[usize],
    ) -> Self {
        if options.parallel {
            #[cfg(not(feature = "rayon"))]
            panic!("Parallel processing requires the `rayon` feature to be enabled.");

            #[cfg(feature = "rayon")]
            Self::for_words_with_count_table_rayon(words, options, word_counts)
        } else {
            Self::for_words_with_count_table_serial(words, options, word_counts)
        }
    }

    /// Build a [`PairIndex`] from a slice of [`Word`]s, using a count table.
    ///
    /// This is a serial implementation that does not use parallelism;
    /// this ignores the `options.parallel` flag.
    ///
    /// # Arguments
    /// * `words` - the slice of words.
    /// * `options` - options for building the index.
    /// * `word_counts` - `word_counts[i]` is the duplication count of `words[i]`.
    pub fn for_words_with_count_table_serial(
        words: &[Word<T>],
        _options: PairIndexOptions,
        word_counts: &[usize],
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
    /// * `words` - the slice of words.
    /// * `options` - options for building the index.
    /// * `word_counts` - `word_counts[i]` is the duplication count of `words[i]`.
    #[cfg(feature = "rayon")]
    pub fn for_words_with_count_table_rayon(
        words: &[Word<T>],
        _options: PairIndexOptions,
        word_counts: &[usize],
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
            Word::from(['h' as u8, 'e' as u8, 'l' as u8, 'l' as u8, 'o' as u8]),
            Word::from(['w' as u8, 'o' as u8, 'r' as u8, 'l' as u8, 'd' as u8]),
            Word::from(['h' as u8, 'e' as u8, 'l' as u8, 'p' as u8]),
        ];

        for parallel in [true, false] {
            let options = PairIndexOptions { parallel };

            let index = PairIndex::for_words(&words, options);

            let PairIndex {
                pair_counts,
                pair_to_word_index,
            } = index;

            let mut pair_counts: Vec<_> = pair_counts.into_iter().collect();
            pair_counts.sort();
            assert_eq!(
                pair_counts,
                vec![
                    (('e' as u8, 'l' as u8), 2),
                    (('h' as u8, 'e' as u8), 2),
                    (('l' as u8, 'd' as u8), 1),
                    (('l' as u8, 'l' as u8), 1),
                    (('l' as u8, 'o' as u8), 1),
                    (('l' as u8, 'p' as u8), 1),
                    (('o' as u8, 'r' as u8), 1),
                    (('r' as u8, 'l' as u8), 1),
                    (('w' as u8, 'o' as u8), 1),
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
