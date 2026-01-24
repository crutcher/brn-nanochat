//! # Byte/Token Mapping Table

use crate::types::TokenType;
use crate::vocab::TokenVocabIndex;
use crate::vocab::tooling::permutations::{invert_permutation, try_check_permutation};
use ahash::AHashMap;
use core::fmt::Debug;

/// 0..255 Rank Byte/Token Bijection Table
///
/// This will always have 255 entries, one for each byte value.
/// The token values are not required to be dense, or in the range 0..255.
/// This is required to be a bijection (255 distinct tokens).
#[derive(Clone, PartialEq)]
pub struct ByteTable<T: TokenType> {
    /// Hash map from token to byte ordinal value.
    token_to_byte: AHashMap<T, u8>,

    /// Table mapping from byte ordinal (position) to token.
    byte_to_token: [T; 256],
}

impl<T: TokenType> Debug for ByteTable<T> {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        f.debug_struct("ByteTable")
            .field("max_token", &self.max_token())
            .field("tokens", &self.token_to_byte)
            .finish()
    }
}

impl<T: TokenType> Default for ByteTable<T> {
    fn default() -> Self {
        let ord_table = (0..256)
            .map(|i| T::from_usize(i).unwrap())
            .collect::<Vec<_>>();
        Self::from_byte_to_token(&ord_table)
    }
}

impl<T: TokenType> ByteTable<T> {
    /// Construct a new table from a valid permutation.
    ///
    /// # Panics
    /// If the permutation is invalid.
    pub fn from_permutation<P>(perm: P) -> Self
    where
        P: AsRef<[u8]>,
    {
        let perm = perm.as_ref();

        assert_eq!(
            perm.len(),
            256,
            "ByteTable::from_permutation: invalid permutation length"
        );
        let perm: [u8; 256] = perm.try_into().unwrap();

        try_check_permutation(&perm).expect("ByteTable::from_permutation: invalid permutation");

        let inv: [u8; 256] = invert_permutation(&perm).try_into().unwrap();

        let ord_table: [T; 256] = perm
            .iter()
            .map(|&byte| T::from_u8(byte).unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let tokens: AHashMap<T, u8> = inv
            .iter()
            .enumerate()
            .map(|(t, &b)| (T::from_usize(t).unwrap(), b))
            .collect();

        Self {
            token_to_byte: tokens,
            byte_to_token: ord_table,
        }
    }

    /// Build a `ByteTable` from a byte-ord => token table.
    ///
    /// # Panics
    /// If the map is not a 1:1 bijection.
    pub fn from_byte_to_token(token_table: &[T]) -> Self {
        assert_eq!(token_table.len(), 256);

        let ord_table: [T; 256] = token_table.try_into().unwrap();

        let tokens: AHashMap<T, u8> = ord_table
            .iter()
            .enumerate()
            .map(|(t, &token)| (token, t as u8))
            .collect();

        assert_eq!(tokens.len(), 256);

        Self {
            token_to_byte: tokens,
            byte_to_token: ord_table,
        }
    }

    /// Build a `ByteTable` from a token => byte hash map.
    ///
    /// # Panics
    /// If there the map is not a 1:1 bijection.
    pub fn from_token_to_byte(tokens: &AHashMap<T, u8>) -> Self {
        let tokens = tokens.clone();

        let ord_map: AHashMap<u8, T> = tokens.iter().map(|(&t, &b)| (b, t)).collect();
        assert_eq!(ord_map.len(), 256);

        let mut ord_items = ord_map.into_iter().collect::<Vec<_>>();
        ord_items.sort_by_key(|(b, _)| *b);

        let ord_table: [T; 256] = ord_items
            .into_iter()
            .map(|(_, t)| t)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            byte_to_token: ord_table,
            token_to_byte: tokens,
        }
    }

    /// Get the byte-ord => token mapping table.
    pub fn byte_to_token(&self) -> &[T; 256] {
        &self.byte_to_token
    }

    /// Get the token->byte hash map.
    pub fn token_to_byte(&self) -> &AHashMap<T, u8> {
        &self.token_to_byte
    }

    /// Get the token corresponding to a given byte.
    pub fn get_token(
        &self,
        byte: u8,
    ) -> T {
        self.byte_to_token[byte as usize]
    }

    /// Get the byte corresponding to a given token, if any.
    pub fn get_byte(
        &self,
        token: T,
    ) -> Option<u8> {
        self.token_to_byte.get(&token).copied()
    }
}

impl<T: TokenType> TokenVocabIndex<T> for ByteTable<T> {
    fn unordered_tokens_iter(&self) -> impl Iterator<Item = T> {
        (0..256).map(move |i| self.get_token(i as u8))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_table_default() {
        type T = u32;
        let table: ByteTable<T> = ByteTable::default();

        for idx in 0..256 {
            let byte = idx as u8;
            let token = idx as u32;

            assert_eq!(table.get_token(byte), token);
            assert_eq!(table.get_byte(token), Some(byte));
        }
    }

    #[test]
    fn test_byte_table() {
        let mut perm = (0..256).into_iter().map(|i| i as u8).collect::<Vec<_>>();
        perm.reverse();

        type T = u32;
        let table: ByteTable<T> = ByteTable::from_permutation(&perm);

        assert_eq!(table.get_token(0_u8), 255_u32);
        assert_eq!(table.get_token(1_u8), 254_u32);
        assert_eq!(table.get_byte(255_u32), Some(0_u8));
        assert_eq!(table.get_byte(254_u32), Some(1_u8));

        assert_eq!(table.get_byte(256_u32), None);
    }
}
