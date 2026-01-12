//! # Chunk Pair Scan Tokenizer

use crate::DEFAULT_PARALLEL;
use crate::decoders::dictionary_decoder::DictionaryDecoder;
use crate::encoders::TokenEncoder;
use crate::types::TokenType;
use crate::util::regex::regex_pool::RegexWrapperPool;
use crate::util::regex::regex_wrapper::RegexWrapper;
use crate::util::validators;
use crate::vocab::unified_vocab::UnifiedTokenVocab;
use crate::vocab::vocab_index::TokenVocabIndex;
use std::sync::Arc;

/// Config options for the [`UnifiedVocabEncoder`].
#[derive(Debug, Clone)]
pub struct UnifiedVocabEncoderOptions {
    /// Whether to use parallel processing for indexing; requires the `rayon` feature to be enabled.
    pub parallel: bool,
}

impl UnifiedVocabEncoderOptions {
    /// Sets whether to use parallel processing for indexing; requires the `rayon` feature to be enabled.
    pub fn with_parallel(
        self,
        parallel: bool,
    ) -> Self {
        Self {
            parallel: validators::expect_parallel(parallel),
        }
    }
}

impl Default for UnifiedVocabEncoderOptions {
    fn default() -> Self {
        Self {
            parallel: DEFAULT_PARALLEL,
        }
    }
}

/// A Chunk/Pair Scanning [`TokenEncoder`].
#[derive(Clone)]
pub struct UnifiedVocabEncoder<T: TokenType> {
    /// Data for the encoders.
    pub data: Arc<UnifiedTokenVocab<T>>,

    /// Tokenizer options.
    pub options: UnifiedVocabEncoderOptions,

    regex_pool: RegexWrapperPool,
}

impl<T: TokenType> UnifiedVocabEncoder<T> {
    /// Construct a new encoders..
    pub fn new(
        data: Arc<UnifiedTokenVocab<T>>,
        options: UnifiedVocabEncoderOptions,
    ) -> Self {
        #[cfg(not(feature = "rayon"))]
        if options.parallel {
            panic!("Parallel processing requires the `rayon` feature to be enabled.");
        }

        let word_regex: RegexWrapper = data.word_pattern.compile().unwrap();
        let regex_pool = RegexWrapperPool::new(word_regex.into());
        Self {
            data,
            options,
            regex_pool,
        }
    }

    /// Append a chunk of text into token IDs.
    pub fn append_encode_chunk(
        &self,
        buf: &mut Vec<T>,
        chunk: &[u8],
    ) {
        if chunk.len() == 1 {
            buf.push(T::from_u8(chunk[0]).unwrap());
            return;
        }

        if let Some(token) = self.data.word_vocab.lookup_token(chunk) {
            buf.push(token);
            return;
        }

        // Reuse the output buffer as a stack.
        // Append the byte-tokens to the buffer.
        let start = buf.len();
        buf.extend(chunk.iter().map(|&b| T::from_u8(b).unwrap()));

        // Incrementally shrink the "stack" (the new buffer end)
        // Until we can no longer find pairs to merge.
        let stop = start + 2;
        while buf.len() >= stop {
            // Find the pair which merges to the lowest ranked token.
            let mut best_match: Option<(usize, T)> = None;

            for idx in start..buf.len() - 1 {
                let pair = (buf[idx], buf[idx + 1]);

                if let Some(&merge_token) = self.data.pair_vocab.pairs.get(&pair)
                    && (best_match.is_none() || (merge_token < best_match.unwrap().1))
                {
                    best_match = Some((idx, merge_token));
                }
            }

            if let Some((idx, merge_token)) = best_match {
                // buf[idx..=idx+1] (a, b) -> buf[idx] t
                buf[idx] = merge_token;
                buf.remove(idx + 1);
            } else {
                // No more merges possible
                break;
            }
        }
    }

    /// Build a [`TokenDecoder`] from this [`UnifiedVocabEncoder`].
    pub fn to_decoder(&self) -> DictionaryDecoder<T> {
        self.data.to_decoder()
    }
}

impl<T: TokenType> TokenEncoder<T> for UnifiedVocabEncoder<T> {
    fn max_token(&self) -> T {
        self.data.max_token()
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, text)))]
    fn encode<S: AsRef<str>>(
        &self,
        text: S,
    ) -> Vec<T> {
        let text = text.as_ref();
        let mut tokens = Vec::with_capacity(text.len());

        for chunk in self.regex_pool.get().find_iter(text).map(|m| m.as_str()) {
            self.append_encode_chunk(&mut tokens, chunk.as_bytes());
        }
        tokens
    }

    /// Encode a batch of text into tokens.
    fn encode_batch(
        &self,
        batch: &[String],
    ) -> Vec<Vec<T>> {
        if self.options.parallel {
            #[cfg(not(feature = "rayon"))]
            panic!("Parallel processing requires the `rayon` feature to be enabled.");

            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;
                batch.par_iter().map(|text| self.encode(text)).collect()
            }
        } else {
            batch.iter().map(|text| self.encode(text)).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::decoders::token_decoder::TokenDecoder;
    use crate::encoders::TokenEncoder;
    use crate::encoders::unified_encoder::UnifiedVocabEncoder;
    use crate::training::trainer::{BPETokenVocabTrainer, TrainResults};
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::unified_vocab::UnifiedTokenVocab;
    use compact_str::CompactString;
    use std::sync::Arc;

    #[test]
    #[cfg(feature = "rayon")]
    fn test_tokenizer_parallel() {
        test_tokenizer(true);
    }

    #[test]
    fn test_tokenizer_serial() {
        test_tokenizer(false);
    }

    fn test_tokenizer(parallel: bool) {
        type T = u16;
        type C = u32;
        type K = CompactString;

        let options = BPETokenVocabTrainer::new_with_vocab_size(1000).with_parallel(parallel);

        let samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let TrainResults {
            word_pattern,
            pair_vocab,
        } = options
            .train_vocab_from_sample_iter::<T, K, C, _>(samples.iter())
            .unwrap();

        let vocab: Arc<UnifiedTokenVocab<T>> = UnifiedTokenVocab::new(word_pattern.into())
            .with_pair_vocab(pair_vocab)
            .expand_words_from_bpe()
            .into();

        let encoder = UnifiedVocabEncoder::<T>::new(vocab.clone(), Default::default());
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
}
