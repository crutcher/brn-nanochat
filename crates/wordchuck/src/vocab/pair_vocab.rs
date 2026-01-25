//! # Pair Map ``{ (T, T) -> T }`` Token Vocabulary

use crate::decoders::TokenDecoder;
use crate::decoders::pair_decoder::PairExpansionDecoder;
use crate::types::{PairTokenMap, TokenType};
use crate::vocab::byte_table::ByteTokenTable;
use crate::vocab::vocab_index::TokenVocabIndex;
use ahash::AHashSet;
use std::sync::Arc;

/// Validate that a [`ByteTokenTable`] and [`PairTokenMap`] are compatible.
///
/// - for every ``(a, b) -> t`` entry:
///   - the parents ``(a, b)``:
///     - are either in the `byte_table`, or are targets in the map, not both.
///   - the target ``t`` is not in the `byte_table`.
pub fn try_validate_pair_map<T: TokenType>(
    byte_table: &ByteTokenTable<T>,
    pairs: &PairTokenMap<T>,
) -> anyhow::Result<()> {
    let pair_targets: AHashSet<T> = pairs.values().copied().collect();

    for t in &pair_targets {
        if let Some(b) = byte_table.get_byte(*t) {
            anyhow::bail!("Target token in pair map {t:?} also mapped to byte {b:0x?}");
        }
    }

    for (&pair, &t) in pairs.iter() {
        /*
        if pair.0 >= t || pair.1 >= t {
            anyhow::bail!("Illegal pair order: {pair:?} -> {t:?}");
        }
         */

        for pt in [pair.0, pair.1] {
            let is_pair_target = pair_targets.contains(&pt);
            let byte_target = byte_table.get_byte(pt);

            if is_pair_target && let Some(b) = byte_target {
                anyhow::bail!(
                    "Pair {pair:?} -> {t:?} parent {pt:?} is a pair target and byte target: {b:0x?}"
                );
            }
            if !is_pair_target && byte_target.is_none() {
                anyhow::bail!("Pair {pair:?} -> {t:?} parent {pt:?} is not defined");
            }
        }
    }

    Ok(())
}

/// Pair ``(T, T) -> T`` Vocabulary.
///
/// - Grounded in a `ByteTable<T>` for byte-to-token mapping.
/// - Collection of ``(T, T) -> T`` pairs.
#[derive(Default, Debug, Clone)]
pub struct PairTokenMapVocab<T: TokenType> {
    /// Byte/token mapping table.
    byte_table: Arc<ByteTokenTable<T>>,

    /// Map of ``{ (T, T) -> T }``.
    pairs: PairTokenMap<T>,
}

impl<T: TokenType> PairTokenMapVocab<T> {
    /// Initialize a [`PairTokenMapVocab`].
    pub fn init<B>(
        byte_table: B,
        pairs: PairTokenMap<T>,
    ) -> anyhow::Result<Self>
    where
        B: Into<Arc<ByteTokenTable<T>>>,
    {
        let byte_table = byte_table.into();
        try_validate_pair_map(&byte_table, &pairs)?;
        Ok(Self { byte_table, pairs })
    }

    /// Get the byte/token mapping table.
    pub fn byte_table(&self) -> &Arc<ByteTokenTable<T>> {
        &self.byte_table
    }

    /// Get the map of pairs.
    pub fn pairs(&self) -> &PairTokenMap<T> {
        &self.pairs
    }

    /// Get the number of tokens in the vocabulary.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.byte_table.len() + self.pairs.len()
    }

    /// Generate all ``(Vec<u8>, T)`` pairs in the vocabulary.
    ///
    /// This includes the pairs from the `ByteTable`.
    pub fn to_span_pairs(&self) -> impl Iterator<Item = (Vec<u8>, T)> {
        let decoder = PairExpansionDecoder::from_pair_map(self.byte_table().clone(), self.pairs());

        self.byte_table.to_span_pairs().chain(
            self.pairs
                .values()
                .map(move |&t| (decoder.try_decode_to_bytes([t]).unwrap(), t)),
        )
    }
}

impl<T: TokenType> TokenVocabIndex<T> for PairTokenMapVocab<T> {
    fn unordered_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.byte_table
            .unordered_tokens_iter()
            .chain(self.pairs.values().copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokens_sorted() {
        type T = u32;
        let byte_table: Arc<ByteTokenTable<T>> = Arc::new(Default::default());

        let mut vocab = PairTokenMapVocab::<T> {
            pairs: PairTokenMap::default(),
            byte_table: byte_table.clone(),
        };

        assert_eq!(vocab.max_token(), 255);

        assert_eq!(&vocab.sorted_tokens(), &byte_table.sorted_tokens());

        vocab.pairs.insert((1, 2), 300);
        vocab.pairs.insert((3, 4), 301);
        vocab.pairs.insert((300, 301), 302);

        assert_eq!(vocab.max_token(), 302);

        assert_eq!(
            &vocab.sorted_tokens(),
            &byte_table
                .sorted_tokens()
                .into_iter()
                .chain([300_u32, 301, 302].into_iter())
                .collect::<Vec<T>>()
        );
    }
}
