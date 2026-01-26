//! # Pair Expansion ``{ T -> (T, T) }`` Token Decoder

use crate::decoders::decode_context::TokenDecodeContext;
use crate::decoders::token_decoder::TokenDecoder;
use crate::types::{TokenToPairMap, TokenType};
use crate::vocab::{ByteTokenTable, PairTokenMapVocab, TokenVocabIndex};
use std::sync::Arc;

/// A Pair Expansion ``{ T -> (T, T) }``  [`TokenDecoder`].
#[derive(Clone)]
pub struct PairExpansionDecoder<T: TokenType> {
    /// Byte/token mapping table.
    byte_table: Arc<ByteTokenTable<T>>,

    /// Token to pair mapping.
    token_map: TokenToPairMap<T>,
}

impl<T: TokenType> PairExpansionDecoder<T> {
    /// Creates a new Decoder.
    pub fn new<B>(
        byte_table: B,
        token_map: TokenToPairMap<T>,
    ) -> Self
    where
        B: Into<Arc<ByteTokenTable<T>>>,
    {
        Self {
            byte_table: byte_table.into(),
            token_map,
        }
    }

    /// Build a [`PairExpansionDecoder`] from this [`PairTokenMapVocab`].
    pub fn from_pair_vocab(pair_vocab: &PairTokenMapVocab<T>) -> Self {
        let token_map = pair_vocab
            .pairs()
            .iter()
            .map(|(&pair, &token)| (token, pair))
            .collect();
        Self::new(pair_vocab.byte_table().clone(), token_map)
    }

    /// Get the byte table.
    pub fn byte_table(&self) -> &Arc<ByteTokenTable<T>> {
        &self.byte_table
    }
}

impl<T: TokenType> TokenVocabIndex<T> for PairExpansionDecoder<T> {
    fn unordered_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.token_map.keys().copied()
    }
}

impl<T: TokenType> TokenDecoder<T> for PairExpansionDecoder<T> {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, buf, tokens)))]
    fn incremental_decode(
        &self,
        ctx: &mut TokenDecodeContext<T>,
    ) -> bool {
        while let Some(t) = ctx.stack.pop() {
            if let Some(b) = self.byte_table.get_byte(t) {
                ctx.buf.push(b);
            } else if let Some((a, b)) = self.token_map.get(&t) {
                ctx.stack.push(*b);
                ctx.stack.push(*a);
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
    use crate::encoders::merge_heap_encoder::MergeHeapVocabEncoder;
    use crate::encoders::token_encoder::TokenEncoder;
    use crate::regex::default_regex_supplier;
    use crate::segmentation::SegmentationConfig;
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::UnifiedTokenVocab;
    use crate::vocab::byte_table::ByteTokenTable;
    use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
    use crate::vocab::tooling::testing::build_test_vocab;
    use alloc::sync::Arc;

    #[test]
    fn test_pair_decoder() {
        type T = u16;

        let samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let byte_table: Arc<ByteTokenTable<T>> = Arc::new(Default::default());
        let vocab: Arc<UnifiedTokenVocab<T>> = build_test_vocab(
            byte_table.clone(),
            SegmentationConfig::from_pattern(OA_GPT3_CL100K_WORD_PATTERN),
        )
        .into();

        let encoder = MergeHeapVocabEncoder::<T>::init(vocab.clone(), default_regex_supplier);

        let decoder = PairExpansionDecoder::from_pair_vocab(&vocab.pair_vocab);
        check_is_send(&decoder);
        check_is_sync(&decoder);

        for sample in samples {
            let tokens = encoder.encode(sample);
            let decoded = decoder.try_decode_to_string(&tokens).unwrap();
            assert_eq!(decoded, sample);
        }
    }
}
