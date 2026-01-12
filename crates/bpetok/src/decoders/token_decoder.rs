//! # Token Decoder Trait

use crate::decoders::TokenDecodeContext;
use crate::types::TokenType;
use crate::vocab::TokenVocabIndex;

/// Trait for token decoders.
pub trait TokenDecoder<T: TokenType>: TokenVocabIndex<T> + Send + Sync {
    /// Incrementally decodes the context.
    ///
    /// Progresses until `ctx.stack` is empty,
    /// or the top token cannot be decoded by this decoders.
    ///
    /// # Returns
    /// `ctx.stack.is_empty()`
    fn incremental_decode(
        &self,
        ctx: &mut TokenDecodeContext<T>,
    ) -> bool;

    /// Decodes tokens into bytes.
    ///
    /// # Arguments
    /// * `tokens` - A slice of tokens to decode.
    fn decode_to_context<S: AsRef<[T]>>(
        &self,
        tokens: S,
    ) -> TokenDecodeContext<T> {
        let mut context = TokenDecodeContext::for_tokens(tokens.as_ref().to_vec(), 2);
        self.incremental_decode(&mut context);
        context
    }

    /// Decode tokens into bytes, returning an error if the decoding fails.
    fn try_decode_to_bytes<S: AsRef<[T]>>(
        &self,
        tokens: S,
    ) -> anyhow::Result<Vec<u8>> {
        self.decode_to_context(tokens).try_complete()
    }

    /// Decodes a batch of tokens into a vector of byte vectors, returning an error if the decoding fails.
    fn try_decode_batch_to_bytes(
        &self,
        batch: &[Vec<T>],
    ) -> anyhow::Result<Vec<Vec<u8>>> {
        batch.iter().map(|t| self.try_decode_to_bytes(t)).collect()
    }

    /// Decodes tokens into a string, returning an error if the decoding fails.
    fn try_decode_to_string<S: AsRef<[T]>>(
        &self,
        tokens: S,
    ) -> anyhow::Result<String> {
        Ok(String::from_utf8(self.try_decode_to_bytes(tokens)?)?)
    }

    /// Decodes a batch of tokens into a vector of strings, returning an error if the decoding fails.
    fn try_decode_batch_to_strings(
        &self,
        batch: &[Vec<T>],
    ) -> anyhow::Result<Vec<String>> {
        self.try_decode_batch_to_bytes(batch).map(|b| {
            b.iter()
                .map(|b| String::from_utf8(b.to_vec()).unwrap())
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoders::byte_decoder::ByteDecoder;
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

        let mut ctx = TokenDecodeContext::for_tokens(tokens, 2);
        assert!(!decoder.incremental_decode(&mut ctx));

        assert_eq!(ctx.buf, "hello world".as_bytes().to_vec());
        assert_eq!(ctx.stack, [3000, 256]);
    }

    #[test]
    fn test_decode_to_strings() {
        type T = u32;
        let decoder: ByteDecoder<T> = ByteDecoder::default();

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

        // Test the batch interfaces.
        let string_batch = decoder.try_decode_batch_to_strings(&token_batch).unwrap();
        assert_eq!(string_batch, str_samples);

        // Test the single-sample interfaces.
        for (sample, tokens) in str_samples.iter().zip(token_batch.iter()) {
            assert_eq!(
                decoder.try_decode_to_string(tokens).unwrap(),
                sample.to_string()
            );
        }
    }
}
