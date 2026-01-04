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

    /// Decodes tokens into bytes.
    fn decode_to_bytes<S: AsRef<[T]>>(
        &self,
        tokens: S,
    ) -> Vec<u8>;

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
