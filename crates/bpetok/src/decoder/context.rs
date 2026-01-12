//! # Decoder Context

use crate::types::TokenType;

/// Representation of a token decoding context.
#[derive(Clone)]
pub struct DecodeContext<T: TokenType> {
    /// Append buffer for decoded bytes.
    pub buf: Vec<u8>,

    /// FILO stack of tokens to be decoded.
    pub stack: Vec<T>,
}

impl<T: TokenType> DecodeContext<T> {
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
}
