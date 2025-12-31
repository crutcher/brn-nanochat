//! # Tokenizer Structures

use crate::decoder::TokenDecoder;
use crate::{
    CountType, DEFAULT_PARALLEL, DEFAULT_PATTERN, MergeJob, Pair, PairIndex, PairIndexOptions,
    StringChunkType, TokenType, Word, WordCounter, WordCounterOptions,
};
use ahash::{AHashMap, AHashSet};
use dary_heap::OctonaryHeap;
use fancy_regex::Regex;

/// The size of the u8 space.
pub const U8_SIZE: usize = 256;

fn expect_vocab_size(vocab_size: usize) -> usize {
    assert!(
        vocab_size >= U8_SIZE,
        "vocab_size ({vocab_size}) must be >= 256 (the size of the u8 space)"
    );
    vocab_size
}

fn expect_regex<S: AsRef<str>>(pattern: S) -> Regex {
    let pattern = pattern.as_ref();
    Regex::new(pattern).unwrap_or_else(|_| panic!("regex pattern compilation failed: {}", pattern))
}

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
            vocab_size: expect_vocab_size(vocab_size),
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
            vocab_size: expect_vocab_size(vocab_size),
            ..self
        }
    }

    /// Sets the regex pattern used for text splitting.
    pub fn with_pattern(
        self,
        pattern: impl Into<String>,
    ) -> Self {
        let pattern = pattern.into();
        expect_regex(&pattern);

        Self { pattern, ..self }
    }

    /// Sets whether to use parallel processing for indexing; requires the `rayon` feature to be enabled.
    pub fn with_parallel(
        self,
        parallel: bool,
    ) -> Self {
        Self { parallel, ..self }
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
        expect_vocab_size(self.vocab_size);

        let num_merges = self.vocab_size - U8_SIZE;
        log::info!("Starting BPE training: {} merges to compute", num_merges);

        // Prefer to fail before we do all the work below.
        let compiled_pattern = expect_regex(&self.pattern);

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
            let mut local_pos_updates: AHashMap<Pair<T>, AHashSet<usize>> = AHashMap::new();
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

        log::info!("Finished training: {} merges completed", merges_done);

        Tokenizer {
            merges,
            pattern: self.pattern,
            parallel: self.parallel,
            compiled_pattern,
        }
    }
}

/// A Byte Pair Encoding / Decoding Tokenizer.
#[derive(Debug)]
pub struct Tokenizer<T: TokenType> {
    /// Maps [`Pair<T>`] to [`T`], representing the byte pair encoding merges.
    pub merges: AHashMap<Pair<T>, T>,

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
        U8_SIZE + self.merges.len()
    }

    /// Encode a chunk of text into token IDs.
    pub fn encode_chunk(
        &self,
        chunk: &str,
    ) -> Vec<T> {
        // Convert chunk to bytes then to tokens.
        let mut chunk_tokens: Vec<T> = chunk.bytes().map(|b| T::from_u8(b).unwrap()).collect();

        // Apply merges iteratively
        while chunk_tokens.len() >= 2 {
            // Find the best pair to merge
            let mut best_pair: Option<(usize, Pair<T>, T)> = None;

            for i in 0..chunk_tokens.len() - 1 {
                let pair: Pair<T> = (chunk_tokens[i], chunk_tokens[i + 1]);
                if let Some(&new_id) = self.merges.get(&pair)
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

            #[cfg(feature = "rayon")]
            self.encode_rayon(text)
        } else {
            self.encode_serial(text)
        }
    }

    /// Encode a string into token IDs in parallel.
    ///
    /// Uses parallel processing, ignoring the `parallel` flag.
    pub fn encode_rayon<S: AsRef<str>>(
        &self,
        text: S,
    ) -> Vec<T> {
        use rayon::prelude::*;

        let text = text.as_ref();

        let mut chunks = self
            .compiled_pattern
            .find_iter(text)
            .enumerate()
            .par_bridge()
            .map(|(i, m)| {
                let chunk = m.expect("regex match failed").as_str();
                (i, self.encode_chunk(chunk))
            })
            .collect::<Vec<_>>();

        chunks.sort_by_key(|(i, _)| *i);

        let total_size = chunks.iter().map(|(_, c)| c.len()).sum::<usize>();
        let mut all_tokens: Vec<T> = Vec::with_capacity(total_size);
        for (_, c) in chunks {
            all_tokens.extend(c);
        }
        all_tokens
    }

    /// Encode a string into token IDs serially.
    ///
    /// Uses serial processing, ignoring the `parallel` flag.
    pub fn encode_serial<S: AsRef<str>>(
        &self,
        text: S,
    ) -> Vec<T> {
        let text = text.as_ref();

        self.compiled_pattern
            .find_iter(text)
            .flat_map(|m| {
                let chunk = m.expect("regex match failed").as_str();
                self.encode_chunk(chunk)
            })
            .collect()
    }

    /// Build a [`TokenDecoder`] from this [`Tokenizer`].
    pub fn to_decoder(&self) -> TokenDecoder<T> {
        TokenDecoder::from_merges(&self.merges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use compact_str::CompactString;

    fn check_is_send<S: Send>(_: S) {}
    fn check_is_sync<S: Sync>(_: S) {}

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
        check_is_send(&tokenizer);
        check_is_sync(&tokenizer);

        let decoder = tokenizer.to_decoder();
        check_is_send(&decoder);
        check_is_sync(&decoder);

        for sample in samples {
            assert_eq!(decoder.decode_to_string(tokenizer.encode(sample)), sample)
        }
    }
}
