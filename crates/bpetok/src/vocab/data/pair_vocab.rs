//! # Binary-Pair Encoding Vocabulary Data

use crate::types::{PairToTokenMap, TokenType};
use crate::vocab::data::TokenVocab;
use serde::{Deserialize, Serialize};

/// Token vocabulary as a binary-pair encoding map of ``{ (T, T) -> T }``.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "T: TokenType", deserialize = "T: TokenType"))]
pub struct PairMapTokenVocab<T: TokenType> {
    /// Map of ``{ (T, T) -> T }``.
    pub pairs: PairToTokenMap<T>,
}

impl<T: TokenType> TokenVocab<T> for PairMapTokenVocab<T> {
    fn compound_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.pairs.values().copied()
    }

    fn max_token(&self) -> T {
        self.pairs
            .values()
            .max()
            .copied()
            .unwrap_or(T::from_u8(u8::MAX).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::FromPrimitive;
    use std::collections::HashSet;

    #[test]
    fn test_tokens_iter() {
        type T = u32;
        let byte_tokens: Vec<T> = (0..256).map(|b| T::from_usize(b).unwrap()).collect();

        let mut vocab = PairMapTokenVocab::<T> {
            pairs: PairToTokenMap::default(),
        };

        assert_eq!(vocab.max_token(), 255);

        assert_eq!(vocab.compound_tokens_iter().collect::<Vec<T>>(), vec![]);

        assert_eq!(&vocab.all_tokens_iter().collect::<Vec<T>>(), &byte_tokens);

        vocab.pairs.insert((1, 2), 300);
        vocab.pairs.insert((3, 4), 301);
        vocab.pairs.insert((300, 301), 302);

        assert_eq!(vocab.max_token(), 302);

        let non_byte_tokens: HashSet<T> = [300, 301, 302].iter().copied().collect();

        let mut combined: HashSet<T> = byte_tokens.iter().copied().collect();
        combined.extend(&non_byte_tokens);

        assert_eq!(
            &vocab.compound_tokens_iter().collect::<HashSet<T>>(),
            &non_byte_tokens,
        );

        assert_eq!(&vocab.all_tokens_iter().collect::<HashSet<T>>(), &combined);
    }
}
