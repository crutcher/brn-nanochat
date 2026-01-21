//! Token Decoder Context

use crate::types::TokenType;
use crate::vocab::public::size_hints::EXPECTED_BYTES_PER_TOKEN;

/// Representation of a token decoding context.
#[derive(Clone)]
pub struct TokenDecodeContext<T: TokenType> {
    /// Append buffer for decoded bytes.
    pub buf: Vec<u8>,

    /// FILO stack of tokens to be decoded.
    pub stack: Vec<T>,
}

impl<T: TokenType> From<Vec<T>> for TokenDecodeContext<T> {
    fn from(tokens: Vec<T>) -> Self {
        Self::for_tokens_with_hint(tokens, EXPECTED_BYTES_PER_TOKEN)
    }
}

impl<T: TokenType> TokenDecodeContext<T> {
    /// Creates a new decoding context.
    ///
    /// # Arguments
    /// * `tokens` - the tokens to decode.
    /// * `bytes_per_token_hint` - a hint for the average number of bytes per token,
    ///   used when allocating output buffer space.
    pub fn for_tokens_with_hint(
        tokens: Vec<T>,
        bytes_per_token_hint: f64,
    ) -> Self {
        let capacity = tokens.len() as f64 * bytes_per_token_hint * 1.25;
        let buf = Vec::with_capacity(capacity as usize);
        let mut stack = tokens;
        stack.reverse();
        Self { buf, stack }
    }

    /// The context is complete when the token stack is empty.
    pub fn is_complete(&self) -> bool {
        self.stack.is_empty()
    }

    /// Returns the decoded buffer, or an error if the stack is not empty.
    pub fn try_result(self) -> anyhow::Result<Vec<u8>> {
        if self.is_complete() {
            Ok(self.buf)
        } else {
            Err(anyhow::anyhow!(
                "Incomplete context: [{:?}, ...]",
                self.stack[self.stack.len() - 1]
            ))
        }
    }

    /// Returns the decoded buffer, panics if the stack is not empty.
    pub fn unwrap(self) -> Vec<u8> {
        self.try_result().unwrap()
    }
}
