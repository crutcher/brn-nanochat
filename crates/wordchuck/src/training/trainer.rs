//! # Vocab Trainer

use crate::regex::RegexWrapperPattern;
use crate::training::pair_span_index::PairSpanIndex;
use crate::training::text_span_counter::{TextSpanCounter, TextSpanCounterOptions};
use crate::training::token_span_buffer::TokenSpanBuf;
use crate::types::{CountType, Pair, StringChunkType, TokenType};
use crate::util::validators;
use crate::util::validators::U8_SIZE;
use crate::vocab::UnifiedTokenVocab;
use crate::vocab::byte_table::ByteTable;
use ahash::{AHashMap, AHashSet};
use core::cmp::Ordering;
use dary_heap::OctonaryHeap;
use std::sync::Arc;

/// Options for [`BinaryPairVocabTrainer`].
#[derive(Debug, Clone)]
pub struct BinaryPairVocabTrainerOptions {
    /// The regex pattern used for text splitting.
    pub pattern: RegexWrapperPattern,

    /// The vocab size.
    pub vocab_size: usize,

    /// The byte/token mapping table.
    pub byte_table: ByteTable,
}

impl BinaryPairVocabTrainerOptions {
    /// Create new options.
    pub fn new<P: Into<RegexWrapperPattern>>(
        pattern: P,
        vocab_size: usize,
    ) -> Self {
        Self {
            pattern: pattern.into(),
            vocab_size,
            byte_table: ByteTable::default(),
        }
    }
}

impl BinaryPairVocabTrainerOptions {
    /// Sets the vocab size.
    ///
    /// # Arguments
    /// * `vocab_size` - The desired vocabulary size; must be >= 256 (the size of the u8 space).
    pub fn with_vocab_size(
        self,
        vocab_size: usize,
    ) -> Self {
        Self { vocab_size, ..self }
    }

    /// Sets the regex pattern used for text splitting.
    pub fn with_pattern<P: Into<RegexWrapperPattern>>(
        self,
        pattern: P,
    ) -> Self {
        let pattern = pattern.into();
        pattern.compile().expect("regex pattern compilation failed");
        Self { pattern, ..self }
    }

    /// Sets the byte/token mapping table.
    pub fn with_byte_table(
        self,
        byte_table: ByteTable,
    ) -> Self {
        Self { byte_table, ..self }
    }

    /// Initializes a [`BinaryPairVocabTrainer`] from these options.
    ///
    /// # Parameters
    /// * `K` - the type used to store strings in the word counts.
    /// * `C` - the type used to store counts in the word counts.
    pub fn init<K, C>(self) -> BinaryPairVocabTrainer<K, C>
    where
        K: StringChunkType,
        C: CountType,
    {
        BinaryPairVocabTrainer::init(self)
    }
}

/// Info about a [`Pair`] that could be merged.
#[derive(Debug, Eq)]
pub struct MergeJob<T: TokenType, C: CountType> {
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

/// Trainer for learning binary pair encodings.
///
/// # Parameters
/// * `K` - the type used to store strings in the word counts.
/// * `C` - the type used to store counts in the word counts.
pub struct BinaryPairVocabTrainer<K = String, C = u32>
where
    K: StringChunkType,
    C: CountType,
{
    /// Trainer options.
    pub options: BinaryPairVocabTrainerOptions,

    /// The text span counter.
    pub span_counter: TextSpanCounter<K, C>,
}

impl<K, C> BinaryPairVocabTrainer<K, C>
where
    K: StringChunkType,
    C: CountType,
{
    /// Initializes a [`BinaryPairVocabTrainer`].
    pub fn init(options: BinaryPairVocabTrainerOptions) -> Self {
        let span_counter = TextSpanCounter::<K, C>::new(
            Arc::new(
                options
                    .pattern
                    .compile()
                    .expect("regex pattern compilation failed"),
            ),
            TextSpanCounterOptions::default(),
        );

        BinaryPairVocabTrainer {
            options,
            span_counter,
        }
    }

    /// Update the word counts inplace from a text string.
    pub fn update_from_text<S: AsRef<str>>(
        &mut self,
        text: S,
    ) {
        self.span_counter.update_from_text(text);
    }

    /// Update word counts inplace from a sample iterator.
    pub fn update_from_samples<I>(
        &mut self,
        samples: I,
    ) where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        self.span_counter.update_from_samples(samples);
    }

    /// Trains [`UnifiedTokenVocab<T>`].
    ///
    /// The resulting vocab will contain:
    /// * the trainer's word split pattern,
    /// * a ``{(T, T) -> T}`` pair map vocab with the learned binary pair merges,
    /// * a ``{Vec<u8> -> T}`` word map that is empty.
    ///
    /// # Parameters
    /// * `T` - the [`TokenType`] of the trained vocab.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip(self, words, word_counts))
    )]
    pub fn train<T>(self) -> anyhow::Result<UnifiedTokenVocab<T>>
    where
        T: TokenType,
        C: CountType,
    {
        validators::expect_vocab_size::<T>(self.options.vocab_size);

        let num_merges = self.options.vocab_size - U8_SIZE;
        log::info!("Starting BPE training: {} merges to compute", num_merges);

        self.options.pattern.compile()?;

        let mut vocab = UnifiedTokenVocab::new(self.options.pattern.clone());

        let (mut words, word_counts): (Vec<TokenSpanBuf<T>>, Vec<C>) =
            self.span_counter.to_word_counts_iter().unzip();

        log::info!("Building pair index...");
        let PairSpanIndex {
            mut pair_counts,
            // FIXME(crutcher): should this be updated while we merge?
            pair_index,
        } = PairSpanIndex::from_span_count_table(&words, &word_counts);

        let zero = C::zero();
        let one = C::one();

        // ---- Build heap ----
        log::info!("Building heap with {} unique pairs", pair_counts.len());
        let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
        for (pair, word_indices) in pair_index.into_iter() {
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
            vocab.pair_vocab.add_pair(job.pair, new_token);

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

        vocab.shrink_to_fit();

        log::info!("Finished training: {} merges completed", merges_done);

        Ok(vocab)
    }
}

#[cfg(test)]
mod tests {
    use crate::decoders::token_decoder::TokenDecoder;
    use crate::encoders::token_encoder::TokenEncoder;
    use crate::encoders::unified_encoder::UnifiedVocabEncoder;
    use crate::training::trainer::{BinaryPairVocabTrainerOptions, MergeJob};
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::TokenVocabIndex;
    use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
    use crate::vocab::unified_vocab::UnifiedTokenVocab;
    use alloc::sync::Arc;
    use compact_str::CompactString;
    use core::cmp::Ordering;

    #[test]
    fn test_tokenizer_options() {
        let options = BinaryPairVocabTrainerOptions::new(OA_GPT3_CL100K_WORD_PATTERN, 1000);

        assert_eq!(options.vocab_size, 1000);
        assert_eq!(options.pattern, OA_GPT3_CL100K_WORD_PATTERN.into());

        let options = options.with_vocab_size(2000).with_pattern(r"\S+");

        assert_eq!(options.vocab_size, 2000);
        assert_eq!(options.pattern, r"\S+".into());
    }

    #[test]
    #[should_panic(expected = "regex pattern compilation failed")]
    fn test_tokenizer_options_bad_pattern() {
        let _ = BinaryPairVocabTrainerOptions::new(r"(", 1000).init::<String, u32>();
    }

    #[test]
    fn test_train_tokenizer() {
        type T = u16;
        type C = u32;
        type K = CompactString;

        let options = BinaryPairVocabTrainerOptions::new(OA_GPT3_CL100K_WORD_PATTERN, 1000);

        let samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let mut trainer = options.init::<K, C>();
        trainer.update_from_samples(samples.iter());

        let vocab: Arc<UnifiedTokenVocab<T>> = trainer
            .train::<T>()
            .unwrap()
            .extend_word_vocab_from_pair_vocab()
            .into();

        let encoder = UnifiedVocabEncoder::<T>::new(vocab.clone());
        check_is_send(&encoder);
        check_is_sync(&encoder);

        assert_eq!(encoder.max_token(), 292);

        let decoder = encoder.to_decoder();
        check_is_send(&decoder);
        check_is_sync(&decoder);

        for sample in samples {
            let tokens = encoder.encode(sample);
            assert_eq!(decoder.try_decode_to_string(tokens).unwrap(), sample);
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
