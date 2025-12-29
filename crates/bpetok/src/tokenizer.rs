//! # Tokenizer Structures

use crate::{
    DEFAULT_PARALLEL, DEFAULT_PATTERN, MergeJob, Pair, PairIndex, PairIndexOptions, TokenType,
    Word, WordCounter, WordCounterOptions,
};
use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use dary_heap::OctonaryHeap;
use fancy_regex::Regex;
use std::collections::HashMap;

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
            pattern: DEFAULT_PATTERN.to_string(),
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

    /// Returns the regex pattern as a [`Regex`] object.
    pub fn get_regex(&self) -> Regex {
        Regex::new(&self.pattern).unwrap()
    }

    /// Converts a sample iterator into a word iterator.
    pub fn samples_to_word_counts<T, I, S>(
        &self,
        samples: I,
    ) -> AHashMap<Word<T>, usize>
    where
        T: TokenType,
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
    {
        let mut counter: WordCounter<CompactString, usize> = WordCounter::new(
            WordCounterOptions::default()
                .with_pattern(&self.pattern)
                .with_parallel(self.parallel),
        );
        counter.update_from_samples(samples);

        counter
            .release()
            .into_iter()
            .map(|(word, count)| (Word::from_string(word), count))
            .collect()
    }

    /// Trains a [`Tokenizer`] over a sample iterator.
    pub fn train_from_sample_iterator<T, I, S>(
        self,
        samples: I,
    ) -> Tokenizer<T>
    where
        T: TokenType,
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
    {
        let word_counts = self.samples_to_word_counts(samples);
        self.train_from_word_counts_map(word_counts)
    }

    /// Trains a [`Tokenizer`] over [`Word`]s.
    ///
    /// # Arguments
    /// * `word_counts` - a ``{word: count}`` map.
    pub fn train_from_word_counts_map<T: TokenType>(
        self,
        words: AHashMap<Word<T>, usize>,
    ) -> Tokenizer<T> {
        let mut ws: Vec<Word<T>> = Vec::with_capacity(words.len());
        let mut cs: Vec<usize> = Vec::with_capacity(words.len());

        words.into_iter().for_each(|(w, c)| {
            ws.push(w);
            cs.push(c);
        });

        self.train_from_word_counts_table(ws, &cs)
    }

    /// Trains a [`Tokenizer`] over [`Word`]s.
    ///
    /// # Arguments
    /// * `words` - the words.
    /// * `word_counts` - `word_counts[i]` is the duplication count of `words[i]`.
    pub fn train_from_word_counts_table<T: TokenType>(
        self,
        mut words: Vec<Word<T>>,
        word_counts: &[usize],
    ) -> Tokenizer<T> {
        assert!(
            self.vocab_size >= self.num_reserved,
            "vocab_size must be >= num_reserved: {self:#?}"
        );
        let num_merges = self.vocab_size - 256;
        log::info!("Starting BPE training: {} merges to compute", num_merges);

        let compiled_pattern = self.get_regex();

        let mut merges: HashMap<Pair<T>, T> = HashMap::new();

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
            let Some(mut job) = heap.pop() else {
                break;
            };

            // Lazy refresh
            let current = *pair_counts.get(&job.pair).unwrap_or(&0);
            if job.count != current {
                job.count = current;
                if job.count > 0 {
                    heap.push(job);
                }
                continue;
            }
            if job.count == 0 {
                break;
            }

            // Record merge
            let new_id: T = T::from_usize(256 + merges_done).expect("new_id is a valid T");
            merges.insert(job.pair, new_id);

            // Merge this pair in all words where it occurs
            let mut local_pos_updates: AHashMap<Pair<T>, AHashSet<usize>> = AHashMap::new();
            for &word_idx in &job.word_indices {
                // Apply merge to this word.
                words[word_idx].merge_pair_cb(job.pair, new_id, &mut |pair, delta| {
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
                    job.pair,
                    new_id,
                    job.count
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
pub struct Tokenizer<T: TokenType> {
    /// Maps [`Pair<T>`] to [`T`], representing the byte pair encoding merges.
    pub merges: HashMap<Pair<T>, T>,

    /// The regex pattern used for text splitting.
    pub pattern: String,

    /// The compiled regex pattern.
    compiled_pattern: Regex,
}

impl<T: TokenType> Tokenizer<T> {
    /// Encode a string into token IDs
    pub fn encode(
        &self,
        text: &str,
    ) -> Vec<T> {
        let mut all_ids: Vec<T> = Vec::new();

        // Split text using the regex pattern
        for m in self.compiled_pattern.find_iter(text) {
            let chunk = m.expect("regex match failed").as_str();

            // Convert chunk to bytes then to u32 IDs
            let mut ids: Vec<T> = chunk.bytes().map(|b| T::from_u8(b).unwrap()).collect();

            // Apply merges iteratively
            while ids.len() >= 2 {
                // Find the best pair to merge
                let mut best_pair: Option<(usize, Pair<T>, T)> = None;

                for i in 0..ids.len() - 1 {
                    let pair: Pair<T> = (ids[i], ids[i + 1]);
                    if let Some(&new_id) = self.merges.get(&pair)
                        && (best_pair.is_none() || new_id < best_pair.unwrap().2)
                    {
                        best_pair = Some((i, pair, new_id));
                    }
                }

                // If we found a pair to merge, apply it
                if let Some((idx, _pair, new_id)) = best_pair {
                    ids[idx] = new_id;
                    ids.remove(idx + 1);
                } else {
                    // No more merges possible
                    break;
                }
            }

            all_ids.extend(ids);
        }

        all_ids
    }
}
