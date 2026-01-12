//! # Dictionary Decoder

use crate::decoder::TokenDecoder;
use crate::decoder::expansion_decoder::ExpansionDecoder;
use crate::types::{BinaryPairMap, TokenType};
use crate::vocab::data::BPEMapTokenVocab;
use ahash::AHashMap;
use std::collections::hash_map;

/// A token dictionary [`TokenDecoder<T>`].
#[derive(Clone)]
pub struct DictionaryDecoder<T: TokenType> {
    /// Token to bytes mapping.
    ///
    /// Does not include byte-tokens.
    pub dictionary: AHashMap<T, Vec<u8>>,
}

impl<T: TokenType> DictionaryDecoder<T> {
    /// Creates a new Decoder.
    pub fn new(mut dictionary: AHashMap<T, Vec<u8>>) -> Self {
        dictionary.shrink_to_fit();

        Self { dictionary }
    }

    /// Build a [`DictionaryDecoder`] from a [`TokenDecoder`].
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(decoder)))]
    pub fn from_tokenizer<D: TokenDecoder<T>>(decoder: &D) -> Self {
        let mut dictionary = AHashMap::with_capacity(decoder.max_token().to_usize().unwrap());
        for token in decoder.pair_tokens() {
            dictionary.insert(token, decoder.decode_to_bytes([token]));
        }
        Self::new(dictionary)
    }

    /// Build a [`DictionaryDecoder`] from this [`Tokenizer`].
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(data)))]
    pub fn from_bpe(data: &BPEMapTokenVocab<T>) -> DictionaryDecoder<T> {
        Self::from_merge_map(&data.pairs)
    }

    /// Build a [`DictionaryDecoder`] from this [`Tokenizer`].
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(merges)))]
    pub fn from_merge_map(merges: &BinaryPairMap<T>) -> DictionaryDecoder<T> {
        let gd = ExpansionDecoder::from_pair_map(merges);
        Self::from_tokenizer(&gd)
    }
}

impl<T: TokenType> TokenDecoder<T> for DictionaryDecoder<T> {
    fn pair_tokens(&self) -> impl Iterator<Item = T> {
        self.dictionary.keys().copied()
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
                    .dictionary
                    .get(t)
                    .expect("Token not found in slice map")
                    .as_slice();
                buf.extend_from_slice(slice);
            }
        }
    }

    /// Estimates the memory usage of this decoder.
    fn size_estimate(&self) -> usize {
        size_of::<hash_map::Entry<T, Vec<u8>>>() * self.dictionary.len()
            + self.dictionary.values().map(|v| v.len()).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::TokenEncoder;
    use crate::tokenizer::scanning_encoder::ScanningEncoder;
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::data::unified::UnifiedTokenVocab;
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
            bpe_vocab,
        } = options
            .train_vocab_from_sample_iter::<T, K, C, _>(samples.iter())
            .unwrap();

        let vocab: Arc<UnifiedTokenVocab<T>> = UnifiedTokenVocab::new(word_pattern.into())
            .with_bpe_vocab(bpe_vocab)
            .derive_words()
            .into();

        let encoder = ScanningEncoder::<T>::new(vocab.clone(), Default::default());

        let decoder = DictionaryDecoder::from_bpe(&vocab.bpe_vocab);
        check_is_send(&decoder);
        check_is_sync(&decoder);

        for sample in samples {
            let tokens = encoder.encode(sample);
            let decoded = decoder.decode_to_string(&tokens);
            assert_eq!(decoded, sample);
        }
    }
}
