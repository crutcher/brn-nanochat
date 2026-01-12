//! # Dictionary Decoder

use crate::decoders::decode_context::TokenDecodeContext;
use crate::decoders::token_decoder::TokenDecoder;
use crate::types::{TokenToWordMap, TokenType};
use crate::vocab::TokenVocabIndex;

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

impl<T: TokenType> TokenVocabIndex<T> for DictionaryDecoder<T> {
    fn compound_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.token_to_word.keys().copied()
    }
}

impl<T: TokenType> TokenDecoder<T> for DictionaryDecoder<T> {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, buf, tokens)))]
    fn incremental_decode(
        &self,
        ctx: &mut TokenDecodeContext<T>,
    ) -> bool {
        while let Some(t) = ctx.stack.pop() {
            if let Some(b) = t.to_u8() {
                ctx.buf.push(b);
            } else if let Some(w) = self.token_to_word.get(&t) {
                ctx.buf.extend_from_slice(w.as_slice());
            } else {
                ctx.stack.push(t);
                break;
            }
        }
        ctx.stack.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoders::TokenEncoder;
    use crate::encoders::unified_encoder::UnifiedVocabEncoder;
    use crate::training::trainer::{BPETokenVocabTrainer, TrainResults};
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::unified_vocab::UnifiedTokenVocab;
    use compact_str::CompactString;
    use std::sync::Arc;

    #[test]
    fn test_dictionary_decoder() {
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
            .into();

        let encoder = UnifiedVocabEncoder::<T>::new(vocab.clone(), Default::default());

        let decoder = DictionaryDecoder::new(vocab.compiled_dictionary());
        check_is_send(&decoder);
        check_is_sync(&decoder);

        for sample in samples {
            let tokens = encoder.encode(sample);
            let decoded = decoder.try_decode_to_string(&tokens).unwrap();
            assert_eq!(decoded, sample);
        }
    }
}
