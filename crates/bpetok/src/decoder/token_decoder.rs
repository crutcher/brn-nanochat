//! # Token Decoder Trait

use crate::decoder::DecodeContext;
use crate::types::TokenType;
use crate::vocab::TokenVocabIndex;

/// Trait for token decoders.
pub trait TokenDecoder<T: TokenType>: TokenVocabIndex<T> + Send + Sync {
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
        self.decode_batch_to_bytes(batch)
            .iter()
            .map(|b| String::from_utf8(b.to_vec()).unwrap())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::FromPrimitive;

    /// A decoder that only decodes byte tokens.
    #[derive(Clone)]
    struct ByteDecoder<T: TokenType> {
        _marker: std::marker::PhantomData<T>,
    }

    impl<T: TokenType> ByteDecoder<T> {
        fn new() -> Self {
            Self {
                _marker: std::marker::PhantomData,
            }
        }
    }

    impl<T: TokenType> TokenVocabIndex<T> for ByteDecoder<T> {
        fn compound_tokens_iter(&self) -> impl Iterator<Item = T> {
            vec![].into_iter()
        }
    }

    impl<T: TokenType> TokenDecoder<T> for ByteDecoder<T> {
        #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, buf, tokens)))]
        fn decode_context(
            &self,
            ctx: &mut DecodeContext<T>,
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

    #[test]
    fn test_decode_context() {
        type T = u32;
        let decoder: ByteDecoder<T> = ByteDecoder::new();

        let mut tokens = vec![];
        tokens.extend(
            "hello world"
                .as_bytes()
                .iter()
                .map(|b| T::from_u8(*b).unwrap()),
        );
        tokens.extend_from_slice(&[256, 3000]);

        let mut ctx = DecodeContext::for_tokens(tokens, 2);
        assert!(!decoder.decode_context(&mut ctx));

        assert_eq!(ctx.buf, "hello world".as_bytes().to_vec());
        assert_eq!(ctx.stack, [3000, 256]);
    }

    #[test]
    fn test_decode_batch_to_strings() {
        type T = u32;
        let decoder: ByteDecoder<T> = ByteDecoder::new();

        let str_samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let token_batch: Vec<Vec<T>> = str_samples
            .iter()
            .map(|s| {
                s.as_bytes()
                    .iter()
                    .map(|b| T::from_u8(*b).unwrap())
                    .collect()
            })
            .collect();

        let string_batch = decoder.decode_batch_to_strings(&token_batch);
        assert_eq!(string_batch, str_samples);
    }
}
