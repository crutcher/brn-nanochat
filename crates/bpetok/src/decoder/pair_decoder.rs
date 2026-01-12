//! # Expansion Decoder

use crate::decoder::{DecodeContext, TokenDecoder};
use crate::types::{PairToTokenMap, TokenToPairMap, TokenType};

/// An [`ExpansionMap`] [`TokenDecoder<T>`].
#[derive(Clone)]
pub struct PairExpansionDecoder<T: TokenType> {
    /// Token to pair mapping.
    ///
    /// Does not include byte-tokens.
    pub token_to_pair: TokenToPairMap<T>,
}

impl<T: TokenType> PairExpansionDecoder<T> {
    /// Creates a new Decoder.
    pub fn new(mut token_to_pair: TokenToPairMap<T>) -> Self {
        token_to_pair.shrink_to_fit();
        Self { token_to_pair }
    }

    /// Build a [`PairExpansionDecoder`] from this [`Tokenizer`].
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(merge_map)))]
    pub fn from_pair_map(merge_map: &PairToTokenMap<T>) -> Self {
        let expansion_map = merge_map
            .iter()
            .map(|(&pair, &token)| (token, pair))
            .collect();
        Self::new(expansion_map)
    }
}

impl<T: TokenType> TokenDecoder<T> for PairExpansionDecoder<T> {
    fn compound_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.token_to_pair.keys().copied()
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, buf, tokens)))]
    fn decode_context(
        &self,
        ctx: &mut DecodeContext<T>,
    ) -> bool {
        while let Some(t) = ctx.stack.pop() {
            if let Some(b) = t.to_u8() {
                ctx.buf.push(b);
            } else if let Some((a, b)) = self.token_to_pair.get(&t) {
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
    use crate::tokenizer::TokenEncoder;
    use crate::tokenizer::unified_encoder::ScanningEncoder;
    use crate::training::trainer::{BPETokenVocabTrainer, TrainResults};
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::unified_vocab::UnifiedTokenVocab;
    use compact_str::CompactString;
    use std::sync::Arc;

    #[test]
    fn test_expansion_decoder() {
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

        let decoder = PairExpansionDecoder::from_pair_map(&vocab.pair_vocab.pairs);
        check_is_send(&decoder);
        check_is_sync(&decoder);

        for sample in samples {
            let tokens = encoder.encode(sample);
            let decoded = decoder.decode_to_string(&tokens);
            assert_eq!(decoded, sample);
        }
    }
}
