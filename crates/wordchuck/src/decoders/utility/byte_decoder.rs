//! # Byte Decoder
//!
//! Mainly used for utility.

use crate::decoders::{TokenDecodeContext, TokenDecoder};
use crate::types::TokenType;
use crate::vocab::TokenVocabIndex;
use crate::vocab::byte_table::ByteTokenTable;
use crate::vocab::vocab_index::byte_tokens_iter;
use std::sync::Arc;

/// A decoders that only decodes byte tokens.
#[derive(Clone, Default)]
pub struct ByteDecoder<T: TokenType> {
    byte_table: Arc<ByteTokenTable<T>>,
}

impl<T: TokenType> ByteDecoder<T> {
    /// Create a new byte decoder.
    pub fn new<B>(byte_table: B) -> Self
    where
        B: Into<Arc<ByteTokenTable<T>>>,
    {
        Self {
            byte_table: byte_table.into(),
        }
    }

    /// Get the byte table.
    pub fn byte_table(&self) -> &ByteTokenTable<T> {
        &self.byte_table
    }
}

impl<T: TokenType> TokenVocabIndex<T> for ByteDecoder<T> {
    fn unordered_tokens_iter(&self) -> impl Iterator<Item = T> {
        byte_tokens_iter::<T>()
    }
}

impl<T: TokenType> TokenDecoder<T> for ByteDecoder<T> {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, buf, tokens)))]
    fn incremental_decode(
        &self,
        ctx: &mut TokenDecodeContext<T>,
    ) -> bool {
        while let Some(t) = ctx.stack.pop() {
            if let Some(b) = self.byte_table.get_byte(t) {
                ctx.buf.push(b);
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

    #[test]
    fn test_byte_decoder() {
        type T = u32;
        let decoder: ByteDecoder<T> = ByteDecoder::default();

        assert_eq!(decoder.max_token(), 255);
    }

    #[test]
    fn test_decode_context() {
        type T = u32;
        let decoder: ByteDecoder<T> = ByteDecoder::default();

        let mut tokens = vec![];
        tokens.extend(
            "hello world"
                .as_bytes()
                .iter()
                .map(|&b| decoder.byte_table().get_token(b)),
        );
        tokens.extend_from_slice(&[256, 3000]);

        let mut ctx: TokenDecodeContext<T> = tokens.into();
        assert!(!decoder.incremental_decode(&mut ctx));

        assert_eq!(ctx.buf, "hello world".as_bytes().to_vec());
        assert_eq!(ctx.stack, [3000, 256]);
    }
}
