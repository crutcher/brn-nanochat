//! # Decoder Context

use crate::types::TokenType;

/// Representation of a token decoding context.
#[derive(Clone)]
pub struct TokenDecodeContext<T: TokenType> {
    /// Append buffer for decoded bytes.
    pub buf: Vec<u8>,

    /// FILO stack of tokens to be decoded.
    pub stack: Vec<T>,
}

impl<T: TokenType> TokenDecodeContext<T> {
    /// Creates a new decoding context.
    pub fn for_tokens(
        tokens: Vec<T>,
        size_hint: usize,
    ) -> Self {
        let buf = Vec::with_capacity(tokens.len() * size_hint);
        let mut stack = tokens;
        stack.reverse();
        Self { buf, stack }
    }

    /// Is complete?
    pub fn is_complete(&self) -> bool {
        self.stack.is_empty()
    }

    /// Returns the decoded buffer, or an error if the stack is not empty.
    pub fn try_complete(self) -> anyhow::Result<Vec<u8>> {
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
    pub fn expect_complete(self) -> Vec<u8> {
        self.try_complete().unwrap()
    }
}
