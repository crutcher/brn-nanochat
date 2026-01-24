//! # Byte/Token Mapping Table

use crate::types::TokenType;
use crate::vocab::tooling::permutations::{invert_permutation, try_check_permutation};
use core::fmt::Debug;

/// 0..255 Rank Byte/Token Bijection Table
#[derive(Clone, PartialEq)]
pub struct ByteTable {
    perm: [u8; 256],
    inv: [u8; 256],
}

impl Debug for ByteTable {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        f.debug_struct("ByteTable")
            .field("perm", &self.perm)
            .finish()
    }
}

impl ByteTable {
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

        Self { perm, inv }
    }

    /// Get the token corresponding to a given byte.
    pub fn get_token<T: TokenType>(
        &self,
        byte: u8,
    ) -> T {
        T::from_u8(self.perm[byte as usize]).unwrap()
    }

    /// Get the byte corresponding to a given token, if any.
    pub fn get_byte<T: TokenType>(
        &self,
        token: T,
    ) -> Option<u8> {
        let token_idx = token.to_u8()? as usize;
        Some(self.inv[token_idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_table() {
        let mut perm = (0..256).into_iter().map(|i| i as u8).collect::<Vec<_>>();
        perm.reverse();

        let table = ByteTable::from_permutation(&perm);

        type T = u32;
        assert_eq!(table.get_token::<T>(0_u8), 255_u32);
        assert_eq!(table.get_token::<T>(1_u8), 254_u32);
        assert_eq!(table.get_byte::<T>(255_u32), Some(0_u8));
        assert_eq!(table.get_byte::<T>(254_u32), Some(1_u8));

        assert_eq!(table.get_byte::<T>(256_u32), None);
    }
}
