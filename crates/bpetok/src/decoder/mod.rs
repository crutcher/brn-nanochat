//! # BPE Token Decoder

use crate::TokenType;
pub mod corpus;
pub mod dict;
pub mod graph;

/// Trait for token decoders.
pub trait TokenDecoder<T: TokenType> {
    /// Returns an iterator over the non-byte tokens in this map.
    fn pair_tokens(&self) -> impl Iterator<Item = T>;

    /// Returns the maximum token id in this decoder.
    fn max_token(&self) -> T {
        self.pair_tokens().max().unwrap()
    }

    /// Decode tokens into a byte vector.
    fn decode_append(
        &self,
        buf: &mut Vec<u8>,
        tokens: &[T],
    );

    /// Decodes tokens into bytes.
    fn decode_to_bytes<S: AsRef<[T]>>(
        &self,
        tokens: S,
    ) -> Vec<u8> {
        let tokens = tokens.as_ref();
        let mut buf = Vec::with_capacity(tokens.len() * 2);
        self.decode_append(&mut buf, tokens);
        buf
    }

    /// Decodes tokens into a string.
    fn decode_to_string<S: AsRef<[T]>>(
        &self,
        tokens: S,
    ) -> String {
        let tokens = tokens.as_ref();
        String::from_utf8(self.decode_to_bytes(tokens)).unwrap()
    }

    /// Estimates the memory usage of this decoder.
    ///
    /// Returns a (metadata, buffers) pair.
    fn size_estimate(&self) -> usize;
}
