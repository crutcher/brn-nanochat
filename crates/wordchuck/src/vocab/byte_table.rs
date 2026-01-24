//! # Byte/Token Mapping Table

use crate::types::TokenType;
use crate::vocab::TokenVocabIndex;
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
        let byte_to_token = (0..256)
            .map(|i| T::from_usize(i).unwrap())
            .collect::<Vec<_>>();
        Self::from_byte_to_token(&byte_to_token)
    }
}

impl<T: TokenType> ByteTable<T> {
    /// Build a `ByteTable` from a byte-ord => token table.
    ///
    /// # Panics
    /// If the map is not a 1:1 bijection.
    pub fn from_byte_to_token(byte_to_token: &[T]) -> Self {
        assert_eq!(byte_to_token.len(), 256);

        let byte_to_token: [T; 256] = byte_to_token.try_into().unwrap();

        let token_to_byte: AHashMap<T, u8> = byte_to_token
            .iter()
            .enumerate()
            .map(|(t, &token)| (token, t as u8))
            .collect();

        assert_eq!(token_to_byte.len(), 256);

        Self {
            token_to_byte,
            byte_to_token,
        }
    }

    /// Build a `ByteTable` from a token => byte hash map.
    ///
    /// # Panics
    /// If there the map is not a 1:1 bijection.
    pub fn from_token_to_byte(token_to_byte: &AHashMap<T, u8>) -> Self {
        let token_to_byte = token_to_byte.clone();

        let ord_map: AHashMap<u8, T> = token_to_byte.iter().map(|(&t, &b)| (b, t)).collect();
        assert_eq!(ord_map.len(), 256);

        let mut ord_items = ord_map.into_iter().collect::<Vec<_>>();
        ord_items.sort_by_key(|(b, _)| *b);

        let byte_to_token: [T; 256] = ord_items
            .into_iter()
            .map(|(_, t)| t)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            byte_to_token,
            token_to_byte,
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
        self.byte_to_token.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::FromPrimitive;

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
        type T = u32;
        let byte_to_token: Vec<T> = (0..256)
            .map(|i| T::from_usize(i).unwrap() + 100)
            .collect::<Vec<_>>();

        let table: ByteTable<T> = ByteTable::from_byte_to_token(&byte_to_token);

        assert_eq!(table.get_token(0_u8), 100);
        assert_eq!(table.get_token(255_u8), 355);

        assert_eq!(table.get_byte(99_u32), None);
        assert_eq!(table.get_byte(100_u32), Some(0));
        assert_eq!(table.get_byte(355_u32), Some(255));
        assert_eq!(table.get_byte(356_u32), None);
    }
}
