//! # Token Decoder Trait

use crate::decoder::DecodeContext;
use crate::types::TokenType;
use crate::vocab::byte_tokens_iter;

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

    /// Incrementally decodes the context.
    ///
    /// Progresses until `ctx.stack` is empty,
    /// or the top token cannot be decoded by this decoder.
    ///
    /// # Returns
    /// `ctx.stack.is_empty()`
    fn decode_context(
        &self,
        ctx: &mut DecodeContext<T>,
    ) -> bool;

    /// Decodes tokens into bytes.
    fn decode_to_bytes<S: AsRef<[T]>>(
        &self,
        tokens: S,
    ) -> Vec<u8> {
        let mut context = DecodeContext::for_tokens(tokens.as_ref().to_vec(), 2);
        if !self.decode_context(&mut context) {
            panic!("Failed to decode token: {:?}", context.stack.pop().unwrap());
        }
        context.buf
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
