//! # Expansion Decoder

use crate::decoder::TokenDecoder;
use crate::types::{PairToTokenMap, TokenToPairMap, TokenType};

/// An [`ExpansionMap`] [`TokenDecoder<T>`].
#[derive(Clone)]
pub struct ExpansionDecoder<T: TokenType> {
    /// Token to pair mapping.
    ///
    /// Does not include byte-tokens.
    pub token_to_pair: TokenToPairMap<T>,
}

impl<T: TokenType> ExpansionDecoder<T> {
    /// Creates a new Decoder.
    pub fn new(mut token_to_pair: TokenToPairMap<T>) -> Self {
        token_to_pair.shrink_to_fit();
        Self { token_to_pair }
    }

    /// Build a [`ExpansionDecoder`] from this [`Tokenizer`].
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(merge_map)))]
    pub fn from_pair_map(merge_map: &PairToTokenMap<T>) -> Self {
        let expansion_map = merge_map
            .iter()
            .map(|(&pair, &token)| (token, pair))
            .collect();
        Self::new(expansion_map)
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, buf, tokens)))]
    fn decode_append_stack(
        &self,
        buf: &mut Vec<u8>,
        stack: &mut Vec<T>,
    ) {
        while let Some(t) = stack.pop() {
            if let Some(b) = t.to_u8() {
                buf.push(b);
                continue;
            }
            let (a, b) = self
                .token_to_pair
                .get(&t)
                .expect("Token not found in slice map");
            stack.push(*b);
            stack.push(*a);
        }
    }
}

impl<T: TokenType> TokenDecoder<T> for ExpansionDecoder<T> {
    fn compound_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.token_to_pair.keys().copied()
    }

    /// Decode tokens into a byte vector.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, buf, tokens)))]
    fn decode_append(
        &self,
        buf: &mut Vec<u8>,
        tokens: &[T],
    ) {
        let mut stack: Vec<T> = Vec::with_capacity(tokens.len() * 2);
        stack.extend(tokens.iter().rev());
        self.decode_append_stack(buf, &mut stack);
    }

    /// Decodes tokens into bytes.
    fn decode_to_bytes<S: AsRef<[T]>>(
        &self,
        tokens: S,
    ) -> Vec<u8> {
        let mut stack = tokens.as_ref().to_vec();
        stack.reverse();

        let mut buf = Vec::with_capacity(stack.len() * 4);

        self.decode_append_stack(&mut buf, &mut stack);
        buf
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

        let decoder = ExpansionDecoder::from_pair_map(&vocab.pair_vocab.pairs);
        check_is_send(&decoder);
        check_is_sync(&decoder);

        for sample in samples {
            let tokens = encoder.encode(sample);
            let decoded = decoder.decode_to_string(&tokens);
            assert_eq!(decoded, sample);
        }
    }
}
