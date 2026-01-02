//! # Dictionary Decoder

use crate::{Pair, TokenDecoder, TokenType};
use ahash::AHashMap;
use std::collections::hash_map;

/// A decoder for [`Tokenizer`] decoder with a materialized dictionary.
pub struct DictionaryDecoder<T: TokenType> {
    /// Token to bytes mapping.
    pub dictionary: AHashMap<T, Vec<u8>>,
}

impl<T: TokenType> DictionaryDecoder<T> {
    /// Creates a new Decoder.
    pub fn new(mut dictionary: AHashMap<T, Vec<u8>>) -> Self {
        dictionary.shrink_to_fit();

        Self { dictionary }
    }

    /// Build a [`DictionaryDecoder`] from this [`Tokenizer`].
    pub fn from_merge_map(merges: &AHashMap<Pair<T>, T>) -> DictionaryDecoder<T> {
        let mut dictionary = AHashMap::with_capacity(merges.len());
        for b in 0..crate::validators::U8_SIZE {
            let token = T::from_u8(b as u8).unwrap();
            dictionary.insert(token, vec![b as u8]);
        }

        let token_to_pair = AHashMap::from_iter(merges.iter().map(|(&pair, &token)| (token, pair)));
        for &token in merges.values() {
            Self::materialize_token(&mut dictionary, &token_to_pair, token);
        }

        DictionaryDecoder::new(dictionary)
    }

    fn materialize_token<'a>(
        expansions: &'a mut AHashMap<T, Vec<u8>>,
        token_2_pair: &AHashMap<T, Pair<T>>,
        token: T,
    ) -> &'a Vec<u8> {
        if expansions.contains_key(&token) {
            return expansions.get(&token).unwrap();
        }

        let (a, b) = token_2_pair.get(&token).unwrap().to_owned();

        Self::materialize_token(expansions, token_2_pair, a);
        Self::materialize_token(expansions, token_2_pair, b);
        let abuf = expansions.get(&a).unwrap();
        let bbuf = expansions.get(&b).unwrap();
        let mut buf = Vec::with_capacity(abuf.len() + bbuf.len());
        buf.extend_from_slice(abuf);
        buf.extend_from_slice(bbuf);

        expansions.insert(token, buf);
        expansions.get(&token).unwrap()
    }
}

impl<T: TokenType> TokenDecoder<T> for DictionaryDecoder<T> {
    fn decode_to_bytes<S: AsRef<[T]>>(
        &self,
        tokens: S,
    ) -> Vec<u8> {
        let tokens = tokens.as_ref();

        let mut buf = Vec::with_capacity(tokens.len() * 2);
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

        buf
    }

    /// Estimates the memory usage of this decoder.
    fn size_estimate(&self) -> (usize, usize) {
        (
            size_of::<hash_map::Entry<T, Vec<u8>>>() * self.dictionary.len(),
            self.dictionary.values().map(|v| v.len()).sum::<usize>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Tokenizer, TokenizerOptions};
    use compact_str::CompactString;

    #[test]
    fn test_corpus_decoder() {
        type T = u16;
        type C = u32;
        type K = CompactString;

        let options = TokenizerOptions::with_capacity(1000);

        let samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let tokenizer: Tokenizer<T> =
            options.train_from_sample_iterator::<T, K, C, _>(samples.iter());

        let decoder = DictionaryDecoder::from_merge_map(&tokenizer.merges);

        for sample in samples {
            let tokens = tokenizer.encode(sample);
            let decoded = decoder.decode_to_string(&tokens);
            assert_eq!(decoded, sample);
        }
    }
}
