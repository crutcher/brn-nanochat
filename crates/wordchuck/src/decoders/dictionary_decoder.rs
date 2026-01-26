//! # Dictionary ``{ T -> Vec<u8> }`` Token Decoder

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
    pub fn new(token_to_word: TokenToWordMap<T>) -> Self {
        Self { token_to_word }
    }
}

impl<T: TokenType> TokenVocabIndex<T> for DictionaryDecoder<T> {
    fn unordered_tokens_iter(&self) -> impl Iterator<Item = T> {
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
            if let Some(w) = self.token_to_word.get(&t) {
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
    use crate::encoders::token_encoder::TokenEncoder;
    use crate::encoders::unified_encoder::UnifiedVocabEncoder;
    use crate::training::bpe_trainer::BinaryPairVocabTrainerOptions;
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::byte_table::ByteTokenTable;
    use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
    use crate::vocab::unified_vocab::UnifiedTokenVocab;
    use alloc::sync::Arc;
    use compact_str::CompactString;

    #[test]
    fn test_dictionary_decoder() {
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

        let byte_table: Arc<ByteTokenTable<T>> = Arc::new(Default::default());

        let vocab: Arc<UnifiedTokenVocab<T>> = trainer
            .train(byte_table.clone())
            .expect("training vocab should succeed")
            .into();

        let encoder = UnifiedVocabEncoder::<T>::new(vocab.clone());

        let decoder = DictionaryDecoder::new(vocab.unified_dictionary());
        check_is_send(&decoder);
        check_is_sync(&decoder);

        assert_eq!(decoder.max_token(), vocab.max_token());

        for sample in samples {
            let tokens = encoder.encode(sample);
            let decoded = decoder.try_decode_to_string(&tokens).unwrap();
            assert_eq!(decoded, sample);
        }
    }
}
