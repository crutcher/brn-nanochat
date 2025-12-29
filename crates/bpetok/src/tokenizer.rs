//! # Tokenizer Structures

use crate::{
    DEFAULT_PARALLEL, MergeJob, Pair, PairIndex, PairIndexOptions, Token, Word, word_count_table,
};
use ahash::{AHashMap, AHashSet};
use dary_heap::OctonaryHeap;
use fancy_regex::Regex;
use std::collections::HashMap;

/// Default GPT-4 style regex pattern for splitting text
pub const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

/// Default number of reserved tokens
pub const DEFAULT_NUM_RESERVED: usize = 256;

/// A builder for [`Tokenizer`]s.
#[derive(Debug)]
pub struct TokenizerOptions {
    /// The regex pattern used for text splitting.
    pub pattern: String,

    /// The vocab size.
    pub vocab_size: usize,

    /// The number of reserved tokens.
    pub num_reserved: usize,

    /// Whether to use parallel processing for indexing; requires the `rayon` feature to be enabled.
    pub parallel: bool,
}

impl Default for TokenizerOptions {
    fn default() -> Self {
        Self {
            pattern: GPT4_PATTERN.to_string(),
            vocab_size: 0,
            num_reserved: 256,
            parallel: DEFAULT_PARALLEL,
        }
    }
}

impl TokenizerOptions {
    /// Sets the regex pattern used for text splitting.
    pub fn with_pattern(
        self,
        pattern: impl Into<String>,
    ) -> Self {
        Self {
            pattern: pattern.into(),
            ..self
        }
    }

    /// Sets the vocab size.
    pub fn with_vocab_size(
        self,
        vocab_size: usize,
    ) -> Self {
        Self { vocab_size, ..self }
    }

    /// Sets the number of reserved tokens.
    pub fn with_num_reserved(
        self,
        num_reserved: usize,
    ) -> Self {
        Self {
            num_reserved,
            ..self
        }
    }

    /// Sets whether to use parallel processing for indexing; requires the `rayon` feature to be enabled.
    pub fn with_parallel(
        self,
        parallel: bool,
    ) -> Self {
        Self { parallel, ..self }
    }

    /// Trains a [`Tokenizer`] over a word sequence.
    pub fn train<T: Token>(
        self,
        mut words: Vec<Word<T>>,
    ) -> Tokenizer<T> {
        assert!(
            self.vocab_size >= self.num_reserved,
            "vocab_size must be >= num_reserved: {self:#?}"
        );
        let num_merges = self.vocab_size - 256;
        log::info!("Starting BPE training: {} merges to compute", num_merges);

        let word_counts = word_count_table(&words);

        let mut merges: HashMap<Pair<T>, T> = HashMap::new();
        let compiled_pattern = Regex::new(&self.pattern).unwrap();

        log::info!("Building pair index...");
        let PairIndex {
            mut pair_counts,
            pair_to_word_index,
        } = PairIndex::for_words_with_count_table(
            &words,
            &word_counts,
            PairIndexOptions {
                parallel: self.parallel,
            },
        );

        // ---- Build heap ----
        log::info!("Building heap with {} unique pairs", pair_counts.len());
        let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
        for (pair, word_indices) in pair_to_word_index.into_iter() {
            let count = *pair_counts.get(&pair).unwrap_or(&0);
            if count > 0 {
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

        while merges_done < num_merges {
            let Some(mut top) = heap.pop() else {
                break;
            };

            // Lazy refresh
            let current = *pair_counts.get(&top.pair).unwrap_or(&0);
            if top.count != current {
                top.count = current;
                if top.count > 0 {
                    heap.push(top);
                }
                continue;
            }
            if top.count == 0 {
                break;
            }

            // Record merge
            let new_id: T = T::from_usize(256 + merges_done).expect("new_id is a valid T");
            merges.insert(top.pair, new_id);

            // Merge this pair in all words where it occurs
            let mut local_pos_updates: AHashMap<Pair<T>, AHashSet<usize>> = AHashMap::new();
            for &word_idx in &top.word_indices {
                // Apply merge to this word.
                words[word_idx].merge_pair_cb(top.pair, new_id, &mut |pair, delta| {
                    // Update global pair counts based on this word's count
                    if delta < 0 {
                        *pair_counts.entry(pair).or_default() -= 1;
                    }
                    if delta > 0 {
                        *pair_counts.entry(pair).or_default() += 1;
                        local_pos_updates.entry(pair).or_default().insert(word_idx);
                    }
                });
            }

            // Add the updated pair counts back to the heap
            for (pair, word_indices) in local_pos_updates {
                let count = *pair_counts.get(&pair).unwrap_or(&0);
                if count > 0 {
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
                    top.pair,
                    new_id,
                    top.count
                );
                last_log_percent = current_percent;
            }
        }

        log::info!("Finished training: {} merges completed", merges_done);

        Tokenizer {
            merges,
            pattern: self.pattern,
            compiled_pattern,
        }
    }
}

/// A Byte Pair Encoding / Decoding Tokenizer.
#[derive(Debug)]
pub struct Tokenizer<T: Token> {
    /// Maps [`Pair<T>`] to [`T`], representing the byte pair encoding merges.
    pub merges: HashMap<Pair<T>, T>,

    /// The regex pattern used for text splitting.
    pub pattern: String,

    /// The compiled regex pattern.
    #[allow(unused)]
    compiled_pattern: Regex,
}
