//! # BPE Token Decoder
use crate::{Pair, TokenType};
use ahash::AHashMap;

/// A decoder for [`Tokenizer`] decoder with a materialized dictionary.
pub struct TokenDecoder<T: TokenType> {
    /// Token to bytes mapping.
    pub dictionary: AHashMap<T, Vec<u8>>,
}

impl<T: TokenType> TokenDecoder<T> {
    /// Creates a new Decoder.
    pub fn new(dictionary: AHashMap<T, Vec<u8>>) -> Self {
        Self { dictionary }
    }

    /// Build a [`TokenDecoder`] from this [`Tokenizer`].
    pub fn from_merges(merges: &AHashMap<Pair<T>, T>) -> TokenDecoder<T> {
        let mut expansions = AHashMap::with_capacity(merges.len());
        for b in 0..crate::validators::U8_SIZE {
            let token = T::from_u8(b as u8).unwrap();
            expansions.insert(token, vec![b as u8]);
        }

        let token2pair = AHashMap::from_iter(merges.iter().map(|(&pair, &token)| (token, pair)));
        for &token in merges.values() {
            Self::materialize_token(&mut expansions, &token2pair, token);
        }

        TokenDecoder::new(expansions)
    }

    fn materialize_token<'a>(
        expansions: &'a mut AHashMap<T, Vec<u8>>,
        token2pair: &AHashMap<T, Pair<T>>,
        token: T,
    ) -> &'a Vec<u8> {
        if expansions.contains_key(&token) {
            return expansions.get(&token).unwrap();
        }

        let (a, b) = token2pair.get(&token).unwrap().to_owned();

        Self::materialize_token(expansions, token2pair, a);
        Self::materialize_token(expansions, token2pair, b);
        let abuf = expansions.get(&a).unwrap();
        let bbuf = expansions.get(&b).unwrap();
        let mut buf = Vec::with_capacity(abuf.len() + bbuf.len());
        buf.extend_from_slice(abuf);
        buf.extend_from_slice(bbuf);

        expansions.insert(token, buf);
        expansions.get(&token).unwrap()
    }

    /// Decodes tokens into bytes.
    pub fn decode_to_bytes<S: AsRef<[T]>>(
        &self,
        tokens: S,
    ) -> Vec<u8> {
        let tokens = tokens.as_ref();

        let total_size = tokens
            .iter()
            .map(|t| self.dictionary.get(t).unwrap().len())
            .sum::<usize>();
        let mut buf = Vec::with_capacity(total_size);
        for token in tokens {
            buf.extend_from_slice(self.dictionary.get(token).unwrap());
        }
        buf
    }

    /// Decodes tokens into a string.
    pub fn decode_to_string<S: AsRef<[T]>>(
        &self,
        tokens: S,
    ) -> String {
        let tokens = tokens.as_ref();
        String::from_utf8(self.decode_to_bytes(tokens)).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder() {
        let merges = AHashMap::from([
            (('h' as usize, 'e' as usize), 300),
            (('l' as usize, 'l' as usize), 301),
        ]);
        let decoder = TokenDecoder::from_merges(&merges);

        assert_eq!(decoder.decode_to_string(&[300, 301, 'o' as usize]), "hello");
    }
}
