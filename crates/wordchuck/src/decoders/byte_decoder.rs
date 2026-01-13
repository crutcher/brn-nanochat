//! # Byte Decoder
//!
//! Mainly used for testing.

use crate::decoders::{TokenDecodeContext, TokenDecoder};
use crate::types::TokenType;
use crate::vocab::TokenVocabIndex;

/// A decoders that only decodes byte tokens.
#[derive(Clone, Default)]
pub struct ByteDecoder<T: TokenType> {
    _marker: std::marker::PhantomData<T>,
}

impl<T: TokenType> TokenVocabIndex<T> for ByteDecoder<T> {
    fn compound_tokens_iter(&self) -> impl Iterator<Item = T> {
        vec![].into_iter()
    }
}

impl<T: TokenType> TokenDecoder<T> for ByteDecoder<T> {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, buf, tokens)))]
    fn incremental_decode(
        &self,
        ctx: &mut TokenDecodeContext<T>,
    ) -> bool {
        while let Some(t) = ctx.stack.pop() {
            if let Some(b) = t.to_u8() {
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
    use num_traits::FromPrimitive;

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
                .map(|b| T::from_u8(*b).unwrap()),
        );
        tokens.extend_from_slice(&[256, 3000]);

        let mut ctx = TokenDecodeContext::for_tokens(tokens);
        assert!(!decoder.incremental_decode(&mut ctx));

        assert_eq!(ctx.buf, "hello world".as_bytes().to_vec());
        assert_eq!(ctx.stack, [3000, 256]);
    }
}
