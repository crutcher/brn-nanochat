//! # Chunk Pair Scan Tokenizer

use crate::DEFAULT_PARALLEL;
use crate::decoders::dictionary_decoder::DictionaryDecoder;
use crate::encoders::text_segmentor::{TextSegmentor, WordRef};
use crate::encoders::token_encoder::TokenEncoder;
use crate::types::TokenType;
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

    segmentor: TextSegmentor,
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

        let specials = match &data.specials {
            Some(specials) => specials
                .words
                .keys()
                .map(|word| String::from_utf8(word.clone()).unwrap())
                .collect::<Vec<String>>()
                .into(),
            None => None,
        };

        let segmentor = TextSegmentor::new(data.word_pattern.clone(), specials);

        Self {
            data,
            options,
            segmentor,
        }
    }

    /// Build a [`TokenDecoder`] from this [`UnifiedVocabEncoder`].
    pub fn to_decoder(&self) -> DictionaryDecoder<T> {
        self.data.to_decoder()
    }
}

impl<T: TokenType> TokenEncoder<T> for UnifiedVocabEncoder<T> {
    /// Encode a word chunk into token IDs.
    fn encode_append_word(
        &self,
        word: &str,
        tokens: &mut Vec<T>,
    ) {
        let chunk = word.as_bytes();
        if chunk.len() == 1 {
            tokens.push(T::from_u8(chunk[0]).unwrap());
            return;
        }

        if let Some(token) = self.data.word_vocab.lookup_token(chunk) {
            tokens.push(token);
            return;
        }

        // Reuse the output buffer as a stack.
        // Append the byte-tokens to the buffer.
        let start = tokens.len();
        tokens.extend(chunk.iter().map(|&b| T::from_u8(b).unwrap()));

        // Incrementally shrink the "stack" (the new buffer end)
        // Until we can no longer find pairs to merge.
        let stop = start + 2;
        while tokens.len() >= stop {
            // Find the pair which merges to the lowest ranked token.
            let mut best_match: Option<(usize, T)> = None;

            for idx in start..tokens.len() - 1 {
                let pair = (tokens[idx], tokens[idx + 1]);

                if let Some(&merge_token) = self.data.pair_vocab.pairs.get(&pair)
                    && (best_match.is_none() || (merge_token < best_match.unwrap().1))
                {
                    best_match = Some((idx, merge_token));
                }
            }

            if let Some((idx, merge_token)) = best_match {
                // buf[idx..=idx+1] (a, b) -> buf[idx] t
                tokens[idx] = merge_token;
                tokens.remove(idx + 1);
            } else {
                // No more merges possible
                break;
            }
        }
    }

    fn split_text<'a>(
        &self,
        text: &'a str,
    ) -> Vec<WordRef<'a>> {
        self.segmentor.split_words(text)
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, text)))]
    fn encode_append(
        &self,
        text: &str,
        tokens: &mut Vec<T>,
    ) {
        self.split_text(text).into_iter().for_each(|wr| match wr {
            WordRef::Normal(w) => self.encode_append_word(w, tokens),
            WordRef::Special(s) => {
                let token = self
                    .data
                    .specials
                    .as_ref()
                    .unwrap()
                    .lookup_token(s.as_bytes())
                    .unwrap();

                tokens.push(token);
            }
        });
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

impl<T: TokenType> TokenVocabIndex<T> for UnifiedVocabEncoder<T> {
    fn compound_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.data.compound_tokens_iter()
    }
    fn max_token(&self) -> T {
        self.data.max_token()
    }
}

#[cfg(test)]
mod tests {
    use crate::decoders::token_decoder::TokenDecoder;
    use crate::encoders::token_encoder::TokenEncoder;
    use crate::encoders::unified_encoder::UnifiedVocabEncoder;
    use crate::training::trainer::{BPETokenVocabTrainer, TrainResults};
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::TokenVocabIndex;
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

        let mut vocab: UnifiedTokenVocab<T> =
            UnifiedTokenVocab::new(word_pattern.into()).with_pair_vocab(pair_vocab);

        vocab.specials_vocab_mut().add_str_word("<|HI|>", 3000);

        let special_sample = "hello <|HI|> world";

        let encoder = UnifiedVocabEncoder::<T>::new(Arc::new(vocab), Default::default());
        check_is_send(&encoder);
        check_is_sync(&encoder);

        assert_eq!(encoder.max_token(), 292);

        let decoder = encoder.to_decoder();
        check_is_send(&decoder);
        check_is_sync(&decoder);

        // Special handling.
        let tokens = encoder.encode(special_sample);
        assert_eq!(
            decoder.try_decode_to_string(tokens).unwrap(),
            special_sample
        );

        for sample in samples {
            let tokens = encoder.encode(sample);
            assert_eq!(decoder.try_decode_to_string(tokens).unwrap(), sample);
        }
    }
}
