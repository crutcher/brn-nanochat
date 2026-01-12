//! # BPE Token Decoder
//!
//! The currently recommended decoder is [`CorpusDecoder`].
//!
//! ```terminaloutput
//! Timing Samples:
//! - count: 8192
//! - avg size: 4712
//!
//! Timing Encode:
//! - avg: 1.737349ms
//!
//! Timing Decode: GraphDecoder
//! - decoder est bytes: 1566720
//! - avg: 52.626µs
//!
//! Timing Decode: DictionaryDecoder
//! - decoder est bytes: 1860233
//! - avg: 17.907µs
//!
//! Timing Decode: CorpusDecoder
//! - decoder est bytes: 1820714
//! - avg: 17.698µs
//! ```

use crate::types::TokenType;
use crate::vocab::byte_tokens_iter;

pub mod dictionary_decoder;
pub mod expansion_decoder;

/// Trait for token decoders.
pub trait TokenDecoder<T: TokenType>: Send + Sync {
    /// Returns an iterator over all tokens.
    fn all_tokens_iter(&self) -> impl Iterator<Item = T> {
        byte_tokens_iter().chain(self.compound_tokens_iter())
    }

    /// Returns an iterator over the non-byte tokens.
    fn compound_tokens_iter(&self) -> impl Iterator<Item = T>;

    /// Returns the maximum token id in this decoder.
    fn max_token(&self) -> T {
        self.compound_tokens_iter().max().unwrap()
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

    /// Decodes a batch of tokens into a vector of byte vectors.
    fn decode_batch_to_bytes(
        &self,
        batch: &[Vec<T>],
    ) -> Vec<Vec<u8>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            batch
                .into_par_iter()
                .map(|tokens| self.decode_to_bytes(tokens))
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        batch.iter().map(|t| self.decode_to_bytes(t)).collect()
    }

    /// Decodes tokens into a string.
    fn decode_to_string<S: AsRef<[T]>>(
        &self,
        tokens: S,
    ) -> String {
        let tokens = tokens.as_ref();
        String::from_utf8(self.decode_to_bytes(tokens)).unwrap()
    }

    /// Decodes a batch of tokens into a vector of strings.
    fn decode_batch_to_strings(
        &self,
        batch: &[Vec<T>],
    ) -> Vec<String> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            batch
                .into_par_iter()
                .map(|tokens| self.decode_to_string(tokens))
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        batch.iter().map(|t| self.decode_to_string(t)).collect()
    }
}
