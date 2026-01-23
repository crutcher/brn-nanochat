//! # Byte Token Table

use core::fmt::Debug;
use num_traits::{FromPrimitive, PrimInt};

/// Verifies if the given slice represents a valid permutation of integers from 0 to `n-1`.
///
/// # Parameters
/// - `perm`: A slice of elements implementing the `PrimInt` and `Debug` traits. The slice
///   is expected to contain integers ranging from 0 to `n-1`, where `n` is the length of the slice.
///
/// # Returns
/// - `Ok(())`: If the slice is a valid permutation, i.e., each number from 0 to `n-1` appears exactly once.
/// - `Err`: If the slice is not a valid permutation, providing details about the:
///   - Duplicate values present in the slice,
///   - Missing values in the expected range `[0, n-1]`,
///   - And a debug representation of the input slice, `perm`.
///
/// # Type Parameters
/// - `T`: The type of the elements in the slice. It must implement the following traits:
///   - `PrimInt`: To allow arithmetic operations and conversion to `usize`.
///   - `Debug`: To enable debug output when reporting errors.
pub fn try_check_permutation<T>(perm: &[T]) -> anyhow::Result<()>
where
    T: PrimInt + Debug,
{
    let n = perm.len();
    let mut target_counts: Vec<u8> = vec![0; n];

    for &x in perm {
        let target_idx = x.to_usize().unwrap();
        if target_idx >= n {
            anyhow::bail!(
                "Bad {n}-permutation: target {} outside of 0..{}\n{:?}",
                target_idx,
                n,
                perm
            );
        }
        target_counts[target_idx] += 1;
    }

    if target_counts.iter().all(|&x| x == 1) {
        return Ok(());
    }

    // Slow path for nice error messages.

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
///
/// Assumes that the input slice is a valid permutation.
///
/// # Panics
/// If the permutation has target values outside ``0..perm.len()``.
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

        assert_eq!(
            try_check_permutation(&[0, 3, 2]).err().unwrap().to_string(),
            "Bad 3-permutation: target 3 outside of 0..3\n[0, 3, 2]",
        );
    }

    #[test]
    fn test_invert_permutation() {
        assert_eq!(invert_permutation(&[0, 1, 2]), vec![0, 1, 2]);
        assert_eq!(invert_permutation(&[0, 2, 1]), vec![0, 2, 1]);
        assert_eq!(invert_permutation(&[1, 2, 0]), vec![2, 0, 1]);
    }
}
