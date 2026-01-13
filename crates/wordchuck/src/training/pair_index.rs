//! Pair Count / Word Indexing

use crate::training::word::Word;
use crate::types::{CountType, Pair, TokenType};
use ahash::{AHashMap, AHashSet};

/// A map from [`Pair`] to its occurrence count.
pub type PairCountMap<T, C> = AHashMap<Pair<T>, C>;

/// A map from [`Pair`] to indices over ``words``.
pub type PairIndexMap<T> = AHashMap<Pair<T>, AHashSet<usize>>;

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
    pub pair_counts: PairCountMap<T, C>,

    /// A map from [`Pair`] to indices over ``words``.
    pub pair_to_word_index: PairIndexMap<T>,
}

impl<T: TokenType, C: CountType> PairIndex<T, C> {
    /// Build a [`PairIndex`] from a slice of [`Word`]s, using a count table.
    ///
    /// # Arguments
    /// * `words` - the slice of words; Words are assumed to be unique.
    /// * `word_counts` - `word_counts[i]` is the count of `words[i]`.
    /// * `options` - options for building the index.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(words, word_counts)))]
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

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip(pair_counts, pair_to_word_index, index, w, word_count))
    )]
    fn observe_word(
        pair_counts: &mut PairCountMap<T, C>,
        pair_to_word_index: &mut PairIndexMap<T>,
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
        let size_hint = words.len() / 100;
        let mut pair_counts: PairCountMap<T, C> = PairCountMap::with_capacity(size_hint);
        let mut pair_to_word_index: PairIndexMap<T> = PairIndexMap::with_capacity(size_hint);

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
                let mut local_pc: PairCountMap<T, C> = Default::default();
                let mut local_wtu: PairIndexMap<T> = Default::default();
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
    use crate::training::word::Word;

    #[test]
    fn test_pair_index_serial_token_u32_count_usize() {
        test_pair_index::<u32, usize>(false);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_pair_index_rayon_token_u32_count_usize() {
        test_pair_index::<u32, usize>(true);
    }

    #[test]
    fn test_pair_index_serial_token_u16_count_i32() {
        test_pair_index::<u16, i32>(false);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_pair_index_rayon_token_u16_count_i32() {
        test_pair_index::<u16, i32>(true);
    }

    fn test_pair_index<T: TokenType, C: CountType>(parallel: bool) {
        let words: Vec<Word<T>> = vec![
            Word::from_string("hello"),
            Word::from_string("world"),
            Word::from_string("help"),
            Word::from_string("☃"), // "☃" := [0xE2 0x98] 0x83
        ];

        let word_counts: Vec<C> = [1, 2, 3, 4]
            .into_iter()
            .map(|c| C::from_u32(c).unwrap())
            .collect();

        let PairIndex {
            pair_counts,
            pair_to_word_index,
        } = PairIndex::<T, C>::index_unique_word_counts_table(
            &words,
            &word_counts,
            PairIndexOptions::default().with_parallel(parallel),
        );

        assert_eq!(
            pair_counts,
            [
                (('e', 'l'), 4),                   // 1 h[el]lo
                (('h', 'e'), 4),                   // 1 [he]llo, 3 [he]lp
                (('l', 'p'), 3),                   // 3 hel[lp]
                (('l', 'd'), 2),                   // 2 wor[ld]
                (('o', 'r'), 2),                   // 2 w[or]ld
                (('r', 'l'), 2),                   // 2 wo[rl]d
                (('w', 'o'), 2),                   // 2 [wo]rld
                (('l', 'l'), 1),                   // 1 he[ll]o
                (('l', 'o'), 1),                   // 1 hel[lo]
                ((0xE2 as char, 0x98 as char), 4), // "☃" := [0xE2 0x98] 0x83
                ((0x98 as char, 0x83 as char), 4), // "☃" := 0xE2 [0x98 0x83]
            ]
            .into_iter()
            .map(|((a, b), c)| (
                (T::from_u8(a as u8).unwrap(), T::from_u8(b as u8).unwrap()),
                C::from_u32(c).unwrap()
            ))
            .collect::<PairCountMap<T, C>>()
        );

        assert_eq!(
            pair_to_word_index,
            [
                (('e', 'l'), vec![0, 2]),                // "h[el]lo world h[el]p ☃"
                (('h', 'e'), vec![0, 2]),                // "[he]llo world [he]lp ☃"
                (('l', 'd'), vec![1]),                   // "hello wor[ld] help ☃"
                (('l', 'l'), vec![0]),                   // "he[ll]o world help ☃"
                (('l', 'o'), vec![0]),                   // "hel[lo] world help ☃"
                (('l', 'p'), vec![2]),                   // "hello world he[lp] ☃"
                (('o', 'r'), vec![1]),                   // "hello w[or]ld help ☃"
                (('r', 'l'), vec![1]),                   // "hello wo[rl]d help ☃"
                (('w', 'o'), vec![1]),                   // "hello [wo]rld help ☃"
                ((0xE2 as char, 0x98 as char), vec![3]), // "hello world help [☃]"
                ((0x98 as char, 0x83 as char), vec![3]), // "hello world help [☃]"
            ]
            .into_iter()
            .map(|((a, b), s)| (
                (T::from_u8(a as u8).unwrap(), T::from_u8(b as u8).unwrap()),
                AHashSet::from_iter(s)
            ))
            .collect::<PairIndexMap<T>>()
        );
    }
}
