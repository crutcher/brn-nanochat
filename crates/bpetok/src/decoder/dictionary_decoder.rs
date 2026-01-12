//! # Dictionary Decoder

use crate::decoder::TokenDecoder;
use crate::types::{TokenToWordMap, TokenType};

/// A token dictionary [`TokenDecoder<T>`].
#[derive(Clone)]
pub struct DictionaryDecoder<T: TokenType> {
    /// Token to bytes mapping.
    ///
    /// Does not include byte-tokens.
    pub token_to_word: TokenToWordMap<T>,
}

impl<T: TokenType> DictionaryDecoder<T> {
    /// Creates a new Decoder.
    pub fn new(mut token_to_word: TokenToWordMap<T>) -> Self {
        token_to_word.shrink_to_fit();
        Self { token_to_word }
    }
}

impl<T: TokenType> TokenDecoder<T> for DictionaryDecoder<T> {
    fn compound_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.token_to_word.keys().copied()
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, buf, tokens)))]
    fn decode_append(
        &self,
        buf: &mut Vec<u8>,
        tokens: &[T],
    ) {
        for t in tokens {
            if let Some(b) = t.to_u8() {
                buf.push(b);
            } else {
                let slice = self
                    .token_to_word
                    .get(t)
                    .expect("Token not found in slice map")
                    .as_slice();
                buf.extend_from_slice(slice);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::TokenEncoder;
    use crate::tokenizer::unified_encoder::ScanningEncoder;
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::data::unified_vocab::UnifiedTokenVocab;
    use crate::vocab::training::trainer::{BPETokenVocabTrainer, TrainResults};
    use compact_str::CompactString;
    use std::sync::Arc;

    #[test]
    fn test_corpus_decoder() {
        type T = u16;
        type C = u32;
        type K = CompactString;

        let options = BPETokenVocabTrainer::new_with_vocab_size(1000);

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

        let encoder = ScanningEncoder::<T>::new(vocab.clone(), Default::default());

        let decoder = DictionaryDecoder::new(vocab.compiled_dictionary());
        check_is_send(&decoder);
        check_is_sync(&decoder);

        for sample in samples {
            let tokens = encoder.encode(sample);
            let decoded = decoder.decode_to_string(&tokens);
            assert_eq!(decoded, sample);
        }
    }
}
