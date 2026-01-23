//! # Byte Token Table

use core::fmt::Debug;
use num_traits::{FromPrimitive, PrimInt};

/// Checks if a slice is a valid permutation.
pub fn try_check_permutation<T>(perm: &[T]) -> anyhow::Result<()>
where
    T: PrimInt + Debug,
{
    let n = perm.len();
    let mut target_counts: Vec<u8> = vec![0; n];

    for &x in perm {
        target_counts[x.to_usize().unwrap()] += 1;
    }

    if target_counts.iter().all(|&x| x == 1) {
        return Ok(());
    }

    let mut dups: Vec<usize> = Default::default();
    let mut missing: Vec<usize> = Default::default();
    for (target, &count) in target_counts.iter().enumerate() {
        if count > 1 {
            dups.push(target);
        } else if count < 1 {
            missing.push(target);
        }
    }

    anyhow::bail!(
        "Bad {n}-permutation: {:?} duplicated, {:?} missing\n{:?}",
        &dups,
        &missing,
        perm
    )
}

/// Computes the inverse permutation.
pub fn invert_permutation<T>(perm: &[T]) -> Vec<T>
where
    T: PrimInt + FromPrimitive + Debug,
{
    let mut inv = vec![T::zero(); perm.len()];

    for (idx, &target) in perm.iter().enumerate() {
        let inv_idx = target.to_usize().unwrap();
        let inv_target = T::from_usize(idx).unwrap();

        inv[inv_idx] = inv_target;
    }

    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_permutation() {
        assert!(try_check_permutation(&[0, 1, 2]).is_ok());
        assert!(try_check_permutation(&[2, 1, 0]).is_ok());

        assert_eq!(
            try_check_permutation(&[0, 2, 2]).err().unwrap().to_string(),
            "Bad 3-permutation: [2] duplicated, [1] missing\n[0, 2, 2]",
        );
    }

    #[test]
    fn test_invert_permutation() {
        assert_eq!(invert_permutation(&[0, 1, 2]), vec![0, 1, 2]);
        assert_eq!(invert_permutation(&[0, 2, 1]), vec![0, 2, 1]);
        assert_eq!(invert_permutation(&[1, 2, 0]), vec![2, 0, 1]);
    }
}
