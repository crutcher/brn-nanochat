//! # Tokenizer Structures

use crate::decoder::corpus::CorpusDecoder;
use crate::pair_index::{PairIndex, PairIndexOptions};
use crate::validators::U8_SIZE;
use crate::word_count::{WordCounter, WordCounterOptions};
use crate::{
    CountType, DEFAULT_PARALLEL, DEFAULT_PATTERN, Pair, StringChunkType, TokenDecoder, TokenType,
    Word, validators,
};
use ahash::{AHashMap, AHashSet};
use dary_heap::OctonaryHeap;
use fancy_regex::Regex;
use std::cmp::Ordering;

/// A builder for [`Tokenizer`]s.
#[derive(Debug)]
pub struct TokenizerOptions {
    /// The regex pattern used for text splitting.
    pub pattern: String,

    /// The vocab size.
    pub vocab_size: usize,

    /// Whether to use parallel processing for indexing; requires the `rayon` feature to be enabled.
    pub parallel: bool,
}

impl TokenizerOptions {
    /// Creates a new [`TokenizerOptions`].
    ///
    /// # Arguments
    /// * `vocab_size` - The desired vocabulary size; must be >= 256 (the size of the u8 space).
    pub fn with_capacity(vocab_size: usize) -> Self {
        Self {
            pattern: DEFAULT_PATTERN.to_string(),
            vocab_size: validators::expect_vocab_size(vocab_size),
            parallel: DEFAULT_PARALLEL,
        }
    }
}

impl TokenizerOptions {
    /// Sets the vocab size.
    ///
    /// # Arguments
    /// * `vocab_size` - The desired vocabulary size; must be >= 256 (the size of the u8 space).
    pub fn with_vocab_size(
        self,
        vocab_size: usize,
    ) -> Self {
        Self {
            vocab_size: validators::expect_vocab_size(vocab_size),
            ..self
        }
    }

    /// Sets the regex pattern used for text splitting.
    pub fn with_pattern<P: AsRef<str>>(
        self,
        pattern: P,
    ) -> Self {
        let pattern = pattern.as_ref().to_string();
        validators::expect_regex(&pattern);
        Self { pattern, ..self }
    }

    /// Sets whether to use parallel processing for indexing; requires the `rayon` feature to be enabled.
    pub fn with_parallel(
        self,
        parallel: bool,
    ) -> Self {
        Self {
            parallel: validators::expect_parallel(parallel),
            ..self
        }
    }

    /// Validates the options.
    pub fn validate(&self) -> anyhow::Result<()> {
        validators::try_vocab_size(self.vocab_size)?;
        validators::try_regex(&self.pattern)?;
        validators::try_parallel(self.parallel)?;
        Ok(())
    }

    /// Trains a [`Tokenizer`] over a sample iterator.
    pub fn train_from_sample_iterator<T, K, C, I>(
        self,
        samples: I,
    ) -> Tokenizer<T>
    where
        T: TokenType,
        K: StringChunkType,
        C: CountType,
        I: Iterator + Send,
        I::Item: AsRef<str> + Send,
    {
        let word_counts = WordCounter::<K, C>::samples_to_word_counts(
            samples,
            WordCounterOptions::default()
                .with_pattern(&self.pattern)
                .with_parallel(self.parallel),
        );

        self.train_from_word_counts_map(word_counts)
    }

    /// Trains a [`Tokenizer`] over [`Word`]s.
    ///
    /// # Arguments
    /// * `word_counts` - a ``{word: count}`` map.
    pub fn train_from_word_counts_map<T, C>(
        self,
        words: AHashMap<Word<T>, C>,
    ) -> Tokenizer<T>
    where
        T: TokenType,
        C: CountType,
    {
        let (ws, cs): (Vec<Word<T>>, Vec<C>) = words.into_iter().unzip();
        self.train_from_word_counts_table(ws, &cs)
    }

    /// Trains a [`Tokenizer`] over [`Word`]s.
    ///
    /// # Arguments
    /// * `words` - the words.
    /// * `word_counts` - `word_counts[i]` is the duplication count of `words[i]`.
    pub fn train_from_word_counts_table<T, C>(
        self,
        mut words: Vec<Word<T>>,
        word_counts: &[C],
    ) -> Tokenizer<T>
    where
        T: TokenType,
        C: CountType,
    {
        validators::expect_vocab_size(self.vocab_size);

        let num_merges = self.vocab_size - U8_SIZE;
        log::info!("Starting BPE training: {} merges to compute", num_merges);

        // Prefer to fail before we do all the work below.
        let compiled_pattern = validators::expect_regex(&self.pattern);

        let mut merges: AHashMap<Pair<T>, T> = AHashMap::with_capacity(num_merges);

        log::info!("Building pair index...");
        let PairIndex {
            mut pair_counts,
            pair_to_word_index,
        } = PairIndex::index_unique_word_counts_table(
            &words,
            word_counts,
            PairIndexOptions {
                parallel: self.parallel,
            },
        );

        let zero = C::zero();
        let one = C::one();

        // ---- Build heap ----
        log::info!("Building heap with {} unique pairs", pair_counts.len());
        let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
        for (pair, word_indices) in pair_to_word_index.into_iter() {
            let count = *pair_counts.get(&pair).unwrap_or(&zero);
            if count > zero {
                heap.push(MergeJob {
                    pair,
                    count,
                    word_indices,
                });
            }
        }
        // ---- Merge loop ----
        log::info!("Starting merge loop");
        let mut merges_done = 0;
        let mut last_log_percent = 0;

        let mut next_token_index = U8_SIZE;

        while merges_done < num_merges {
            let Some(mut job) = heap.pop() else {
                // No more pairs to merge
                break;
            };

            {
                // Lazy refresh the job count.
                let current = *pair_counts.get(&job.pair).unwrap_or(&zero);
                if job.count != current {
                    job.count = current;
                    if job.count > zero {
                        heap.push(job);
                    }
                    continue;
                }
            }

            if job.count == zero {
                // No live matches.
                break;
            }

            // Generate a new token ID for this merge
            let new_token = T::from_usize(next_token_index).expect("new_token is a valid T");
            next_token_index += 1;

            // Record merge
            merges.insert(job.pair, new_token);

            // Merge this pair in all words where it occurs
            let mut local_pos_updates: AHashMap<Pair<T>, AHashSet<usize>> =
                AHashMap::with_capacity(16);
            for &word_idx in &job.word_indices {
                // Apply merge to this word.
                words[word_idx].merge_pair_cb(job.pair, new_token, &mut |pair, delta| {
                    // Update global pair counts based on this word's count
                    if delta < 0 {
                        *pair_counts.entry(pair).or_default() -= one;
                    }
                    if delta > 0 {
                        *pair_counts.entry(pair).or_default() += one;
                        local_pos_updates.entry(pair).or_default().insert(word_idx);
                    }
                });
            }

            // Add the updated pair counts back to the heap
            for (pair, word_indices) in local_pos_updates {
                let count = *pair_counts.get(&pair).unwrap_or(&zero);
                if count > zero {
                    heap.push(MergeJob {
                        pair,
                        count,
                        word_indices,
                    });
                }
            }

            merges_done += 1;

            // Log progress every 1%
            let current_percent = (merges_done * 100) / num_merges;
            if current_percent > last_log_percent {
                log::info!(
                    "Progress: {}% ({}/{} merges) - Last merge: {:?} -> {:?} (frequency: {})",
                    current_percent,
                    merges_done,
                    num_merges,
                    job.pair,
                    new_token,
                    job.count
                );
                last_log_percent = current_percent;
            }
        }

        merges.shrink_to_fit();

        log::info!("Finished training: {} merges completed", merges_done);

        Tokenizer {
            merge_map: merges,
            pattern: self.pattern,
            parallel: self.parallel,
            compiled_pattern,
        }
    }
}

/// Info about a [`Pair`] that could be merged.
#[derive(Debug, Eq)]
struct MergeJob<T: TokenType, C: CountType> {
    /// The number of instances of this pair in the corpus.
    pub count: C,

    /// The pair to merge.
    pub pair: Pair<T>,

    /// Word indices that may contain this pair.
    pub word_indices: AHashSet<usize>,
}

impl<T: TokenType, C: CountType> MergeJob<T, C> {
    /// The job key.
    ///
    /// Max-heap by count; tie-break to ascending pair order (deterministic)
    pub fn heap_key(&self) -> (C, Pair<T>) {
        (self.count, self.pair)
    }
}

impl<T: TokenType, C: CountType> PartialEq for MergeJob<T, C> {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.heap_key() == other.heap_key()
    }
}

impl<T: TokenType, C: CountType> PartialOrd for MergeJob<T, C> {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: TokenType, C: CountType> Ord for MergeJob<T, C> {
    fn cmp(
        &self,
        other: &Self,
    ) -> Ordering {
        self.heap_key().cmp(&other.heap_key())
    }
}

/// A Byte Pair Encoding / Decoding Tokenizer.
#[derive(Debug)]
pub struct Tokenizer<T: TokenType> {
    /// Maps [`Pair<T>`] to [`T`], representing the byte pair encoding merges.
    pub merge_map: AHashMap<Pair<T>, T>,

    /// The regex pattern used for text splitting.
    pub pattern: String,

    /// Whether to use parallel processing for indexing; requires the `rayon` feature to be enabled.
    pub parallel: bool,

    /// The compiled regex pattern.
    compiled_pattern: Regex,
}

impl<T: TokenType> Tokenizer<T> {
    /// Vocab Size.
    pub fn vocab_size(&self) -> usize {
        U8_SIZE + self.merge_map.len()
    }

    /// Memory usage estimate in bytes.
    pub fn size_estimate(&self) -> usize {
        self.merge_map.capacity() * std::mem::size_of::<Pair<T>>() + self.pattern.len()
    }

    /// Encode a chunk of text into token IDs.
    pub fn encode_chunk<S: AsRef<str>>(
        &self,
        chunk: S,
    ) -> Vec<T> {
        let chunk = chunk.as_ref();

        // Convert chunk to bytes then to tokens.
        let mut chunk_tokens: Vec<T> = chunk.bytes().map(|b| T::from_u8(b).unwrap()).collect();

        // Apply merges iteratively
        while chunk_tokens.len() >= 2 {
            // Find the best pair to merge
            let mut best_pair: Option<(usize, Pair<T>, T)> = None;

            for i in 0..chunk_tokens.len() - 1 {
                let pair: Pair<T> = (chunk_tokens[i], chunk_tokens[i + 1]);
                if let Some(&new_id) = self.merge_map.get(&pair)
                    && (best_pair.is_none() || new_id < best_pair.unwrap().2)
                {
                    best_pair = Some((i, pair, new_id));
                }
            }

            // If we found a pair to merge, apply it
            if let Some((idx, _pair, new_id)) = best_pair {
                chunk_tokens[idx] = new_id;
                chunk_tokens.remove(idx + 1);
            } else {
                // No more merges possible
                break;
            }
        }

        chunk_tokens
    }

    /// Encode a string into token IDs
    pub fn encode<S: AsRef<str>>(
        &self,
        text: S,
    ) -> Vec<T> {
        if self.parallel {
            #[cfg(not(feature = "rayon"))]
            panic!("Parallel processing requires the `rayon` feature to be enabled.");

            // We just fall back to serial, because rayon is currently slower.
        }

        self.encode_serial(text)
    }

    fn split_groups<'a>(
        &'a self,
        text: &'a str,
    ) -> impl Iterator<Item = &'a str> + 'a {
        self.compiled_pattern
            .find_iter(text)
            .map(|m| m.unwrap().as_str())
    }

    /// Encode a string into token IDs in parallel.
    ///
    /// Uses parallel processing, ignoring the `parallel` flag.
    #[cfg(feature = "rayon")]
    pub fn encode_rayon<S: AsRef<str>>(
        &self,
        text: S,
    ) -> Vec<T> {
        use rayon::prelude::*;

        // This is significantly worse?

        self.split_groups(text.as_ref())
            .collect::<Vec<_>>()
            .into_par_iter()
            .flat_map(|chunk| self.encode_chunk(chunk))
            .collect()
    }

    /// Encode a string into token IDs serially.
    ///
    /// Uses serial processing, ignoring the `parallel` flag.
    pub fn encode_serial<S: AsRef<str>>(
        &self,
        text: S,
    ) -> Vec<T> {
        self.split_groups(text.as_ref())
            .flat_map(|chunk| self.encode_chunk(chunk))
            .collect()
    }

    /// Build a [`TokenDecoder`] from this [`Tokenizer`].
    pub fn to_decoder(&self) -> impl TokenDecoder<T> {
        CorpusDecoder::from_merge_map(&self.merge_map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DEFAULT_PARALLEL, DEFAULT_PATTERN, TokenDecoder, types};
    use compact_str::CompactString;

    #[test]
    fn test_tokenizer_options() {
        let options = TokenizerOptions::with_capacity(1000);
        assert_eq!(options.vocab_size, 1000);
        assert_eq!(options.pattern, DEFAULT_PATTERN);
        assert_eq!(options.parallel, DEFAULT_PARALLEL);

        let options = options
            .with_vocab_size(2000)
            .with_pattern(r"\S+")
            .with_parallel(true);

        assert_eq!(options.vocab_size, 2000);
        assert_eq!(options.pattern, r"\S+");
        assert_eq!(options.parallel, true);
    }

    #[test]
    #[should_panic(expected = "vocab_size (255) must be >= 256 (the size of the u8 space)")]
    fn test_tokenizer_options_vocab_size_too_small() {
        let _ = TokenizerOptions::with_capacity(U8_SIZE - 1);
    }

    #[test]
    #[should_panic(expected = "regex pattern compilation failed")]
    fn test_tokenizer_options_bad_pattern() {
        let _ = TokenizerOptions::with_capacity(1000).with_pattern(r"(");
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_train_tokenizer_parallel() {
        test_train_tokenizer(true);
    }

    #[test]
    fn test_train_tokenizer_serial() {
        test_train_tokenizer(false);
    }

    fn test_train_tokenizer(parallel: bool) {
        type T = u16;
        type C = u32;
        type K = CompactString;

        let options = TokenizerOptions::with_capacity(1000).with_parallel(parallel);

        let samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let tokenizer: Tokenizer<T> =
            options.train_from_sample_iterator::<T, K, C, _>(samples.iter());

        // compile time checks.
        types::check_is_send(&tokenizer);
        types::check_is_sync(&tokenizer);

        assert_eq!(tokenizer.vocab_size(), 293);

        let decoder = tokenizer.to_decoder();

        // compile time checks.
        types::check_is_send(&decoder);
        types::check_is_sync(&decoder);

        for sample in samples {
            assert_eq!(decoder.decode_to_string(tokenizer.encode(sample)), sample);
        }
    }

    #[test]
    fn test_merge_job_heap_key() {
        type T = u32;
        type C = u32;

        let job1: MergeJob<T, C> = MergeJob {
            pair: (1, 2),
            count: 2,
            word_indices: Default::default(),
        };

        let job2 = MergeJob {
            pair: (2, 1),
            count: 1,
            word_indices: Default::default(),
        };
        let job3 = MergeJob {
            pair: (2, 2),
            count: 1,
            word_indices: Default::default(),
        };

        assert_eq!(&job1, &job1);
        assert_ne!(&job1, &job2);

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
