//! # Byte Decoder
//!
//! Mainly used for utility.

use crate::decoders::{TokenDecodeContext, TokenDecoder};
use crate::types::TokenType;
use crate::vocab::byte_vocab::ByteVocab;
use std::sync::Arc;

/// A decoders that only decodes byte tokens.
#[derive(Clone, Default)]
pub struct ByteDecoder<T: TokenType> {
    byte_vocab: Arc<ByteVocab<T>>,
}

impl<T: TokenType> ByteDecoder<T> {
    /// Create a new byte decoder.
    pub fn new<B>(byte_vocab: B) -> Self
    where
        B: Into<Arc<ByteVocab<T>>>,
    {
        Self {
            byte_vocab: byte_vocab.into(),
        }
    }

    /// Get the byte table.
    pub fn byte_vocab(&self) -> &ByteVocab<T> {
        &self.byte_vocab
    }
}

impl<T: TokenType> TokenDecoder<T> for ByteDecoder<T> {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, ctx)))]
    fn incremental_decode(
        &self,
        ctx: &mut TokenDecodeContext<T>,
    ) -> bool {
        while let Some(t) = ctx.stack.pop() {
            if let Some(b) = self.byte_vocab.get_byte(t) {
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
    fn test_decode_context() {
        type T = u32;
        let decoder: ByteDecoder<T> = ByteDecoder::default();

        let mut tokens = vec![];
        tokens.extend(
            "hello world"
                .as_bytes()
                .iter()
                .map(|&b| decoder.byte_vocab().get_token(b)),
        );
        tokens.extend_from_slice(&[256, 3000]);

        let mut ctx: TokenDecodeContext<T> = tokens.into();
        assert!(!decoder.incremental_decode(&mut ctx));

        assert_eq!(ctx.buf, "hello world".as_bytes().to_vec());
        assert_eq!(ctx.stack, [3000, 256]);
    }
}
