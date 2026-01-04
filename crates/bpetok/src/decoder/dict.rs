//! # Dictionary Decoder

use crate::data::TokenizerData;
use crate::graph::GraphDecoder;
use crate::{MergeMap, TokenDecoder, TokenType};
use ahash::AHashMap;
use std::collections::hash_map;

/// A decoder for [`Tokenizer`] decoder with a materialized dictionary.
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

    /// Build a [`DictionaryDecoder`] from a [`GraphDecoder`].
    #[tracing::instrument(skip(decoder))]
    pub fn from_tokenizer<D: TokenDecoder<T>>(decoder: &D) -> Self {
        let mut dictionary = AHashMap::with_capacity(decoder.max_token().to_usize().unwrap());
        for token in decoder.pair_tokens() {
            dictionary.insert(token, decoder.decode_to_bytes([token]));
        }
        Self::new(dictionary)
    }

    /// Build a [`DictionaryDecoder`] from this [`Tokenizer`].
    #[tracing::instrument(skip(data))]
    pub fn from_data(data: &TokenizerData<T>) -> DictionaryDecoder<T> {
        Self::from_merge_map(&data.merge_map)
    }

    /// Build a [`DictionaryDecoder`] from this [`Tokenizer`].
    #[tracing::instrument(skip(merges))]
    pub fn from_merge_map(merges: &MergeMap<T>) -> DictionaryDecoder<T> {
        let gd = GraphDecoder::from_merge_map(merges);
        Self::from_tokenizer(&gd)
    }
}

impl<T: TokenType> TokenDecoder<T> for DictionaryDecoder<T> {
    fn pair_tokens(&self) -> impl Iterator<Item = T> {
        self.dictionary.keys().copied()
    }

    #[tracing::instrument(skip(self, buf, tokens))]
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
    use crate::TokenEncoder;
    use crate::builder::TokenizerBuilder;
    use crate::data::TokenizerData;
    use crate::tokenizer::chunkpair::ChunkPairScanTokenizer;
    use crate::types::{check_is_send, check_is_sync};
    use compact_str::CompactString;

    #[test]
    fn test_corpus_decoder() {
        type T = u16;
        type C = u32;
        type K = CompactString;

        let options = TokenizerBuilder::with_capacity(1000);

        let samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let data: TokenizerData<T> =
            options.train_from_sample_iterator::<T, K, C, _>(samples.iter());

        let tokenizer = ChunkPairScanTokenizer::new(data.clone());

        let decoder = DictionaryDecoder::from_data(&data);
        check_is_send(&decoder);
        check_is_sync(&decoder);

        for sample in samples {
            let tokens = tokenizer.encode(sample);
            let decoded = decoder.decode_to_string(&tokens);
            assert_eq!(decoded, sample);
        }
    }
}
