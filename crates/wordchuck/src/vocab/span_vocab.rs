//! # Word Map ``{ Vec<u8> -> T }`` Token Vocabulary

use crate::types::{SpanTokenMap, TokenType};
use crate::vocab::{ByteTokenTable, PairTokenMapVocab, TokenVocabIndex};
use ahash::AHashMap;
use alloc::sync::Arc;

/// Token vocabulary as a dictionary map of ``{ Vec<u8> -> T }``.
#[derive(Debug, Clone, PartialEq)]
pub struct SpanTokenVocab<T: TokenType> {
    /// The byte/token mapping table.
    byte_table: Arc<ByteTokenTable<T>>,

    /// The regex pattern used for text spl
    /// Map of ``{ Vec<u8> -> T }``.
    span_map: SpanTokenMap<T>,
}

impl<T: TokenType> Default for SpanTokenVocab<T> {
    fn default() -> Self {
        SpanTokenVocab::from_byte_table(Arc::new(ByteTokenTable::default()))
    }
}

impl<T: TokenType> From<SpanTokenMap<T>> for SpanTokenVocab<T> {
    fn from(span_map: SpanTokenMap<T>) -> Self {
        Self::from_span_map(span_map)
    }
}

impl<'a, T: TokenType> IntoIterator for &'a SpanTokenVocab<T> {
    type Item = (&'a Vec<u8>, &'a T);

    type IntoIter = std::collections::hash_map::Iter<'a, Vec<u8>, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.span_map.iter()
    }
}

/// Read the ``{ u8 -> T }`` mapping from a ``{ Vec<u8> -> T }`` mapping.
pub fn byte_map_from_span_map<T: TokenType>(span_map: &SpanTokenMap<T>) -> AHashMap<u8, T> {
    span_map
        .iter()
        .filter_map(|(span, &token)| {
            if span.len() == 1 {
                Some((span[0], token))
            } else {
                None
            }
        })
        .collect()
}

/// Validate that a [`ByteTokenTable`] and [`SpanTokenVocab`] are compatible.
pub fn try_validate_span_map<T>(
    byte_table: &ByteTokenTable<T>,
    span_map: &SpanTokenMap<T>,
) -> anyhow::Result<()>
where
    T: TokenType,
{
    for (span, token) in byte_table.span_pairs() {
        let b = span[0];

        if let Some(&map_token) = span_map.get(&span)
            && token != map_token
        {
            anyhow::bail!(
                "ByteTable disagrees with span_map for {b:0x?}: {:?} != {:?}",
                token,
                map_token
            );
        }
    }

    Ok(())
}

impl<T: TokenType> SpanTokenVocab<T> {
    /// Build vocabulary from just a [`ByteTokenTable`].
    ///
    /// Will have 255 span entries, each 1-byte long.
    pub fn from_byte_table<B>(byte_table: B) -> Self
    where
        B: Into<Arc<ByteTokenTable<T>>>,
    {
        let byte_table = byte_table.into();

        let span_map: SpanTokenMap<T> = byte_table.span_pairs().collect();

        Self::init(byte_table, span_map).unwrap()
    }

    /// Build a [`Self`] from a [`SpanTokenMap`].
    ///
    /// The [`ByteTokenTable`] will be inferred from the [`SpanTokenMap`],
    /// and the default ordinal byte to token mappings.
    ///
    /// # Panics
    /// If the [`ByteTokenTable`] mapping is not 1:1.
    pub fn from_span_map(span_map: SpanTokenMap<T>) -> Self {
        let mut byte_map: AHashMap<u8, T> = byte_map_from_span_map(&span_map);
        for ord in 0..256 {
            let b = ord as u8;
            let token = T::from_usize(ord).unwrap();
            if !byte_map.contains_key(&b) {
                byte_map.insert(b, token);
            }
        }

        let mut ord_table: Vec<(u8, T)> = byte_map.into_iter().collect();
        ord_table.sort_by_key(|&(k, _)| k);
        let byte_to_token: Vec<T> = ord_table.into_iter().map(|(_, v)| v).collect();

        let byte_table: Arc<ByteTokenTable<T>> =
            ByteTokenTable::from_byte_to_token(&byte_to_token).into();

        Self::init(byte_table, span_map).unwrap()
    }

    /// Build word vocabulary from a [`PairTokenMapVocab<T>`].
    pub fn from_pair_vocab(pair_vocab: &PairTokenMapVocab<T>) -> Self {
        let byte_table: Arc<ByteTokenTable<T>> = pair_vocab.byte_table().clone();
        let span_map: SpanTokenMap<T> = pair_vocab.span_pairs().collect();

        Self::init(byte_table, span_map).unwrap()
    }

    /// Initialize a [`SpanTokenVocab`].
    ///
    /// The span map will be the union of the span map,
    /// and all overrides from the `byte_table`.
    pub fn init<B>(
        byte_table: B,
        mut span_map: SpanTokenMap<T>,
    ) -> anyhow::Result<Self>
    where
        B: Into<Arc<ByteTokenTable<T>>>,
    {
        let byte_table = byte_table.into();
        try_validate_span_map(&byte_table, &span_map)?;

        span_map.extend(byte_table.span_pairs());

        Ok(Self {
            byte_table,
            span_map,
        })
    }

    /// Get the byte/token mapping table.
    pub fn byte_table(&self) -> &Arc<ByteTokenTable<T>> {
        &self.byte_table
    }

    /// Get the span => token map.
    pub fn span_map(&self) -> &SpanTokenMap<T> {
        &self.span_map
    }

    /// Iterate over the span => token pairs.
    pub fn span_pairs(&self) -> impl Iterator<Item = (Vec<u8>, T)> {
        self.span_map
            .iter()
            .map(|(chunk, &token)| (chunk.clone(), token))
    }

    /// The number of words in the vocabulary.
    pub fn len(&self) -> usize {
        self.span_map.len()
    }

    /// Returns `true` if the vocabulary contains no words.
    pub fn is_empty(&self) -> bool {
        self.span_map.is_empty()
    }

    /// Iterate over the words in the vocabulary.
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a Vec<u8>, &'a T)> + 'a {
        self.span_map.iter()
    }

    /// Return the associated token for the word, if any.
    pub fn lookup_token(
        &self,
        chunk: &[u8],
    ) -> Option<T> {
        if chunk.len() == 1 {
            Some(self.byte_table.get_token(chunk[0]))
        } else {
            self.span_map.get(chunk).copied()
        }
    }

    /// Build a binary pair map from the word vocabulary.
    pub fn to_pair_vocab(&self) -> PairTokenMapVocab<T> {
        let byte_table = self.byte_table.clone();

        let mut pairs = AHashMap::default();

        let token_to_span: AHashMap<T, &[u8]> = self
            .span_map
            .iter()
            .map(|(chunk, &token)| (token, chunk.as_ref()))
            .collect();

        for token in self.unordered_tokens_iter() {
            let span = token_to_span[&token];
            if span.len() <= 1 {
                continue;
            }
            for p in 1..span.len() {
                let pre = &span[..p];
                let post = &span[p..];

                if let Some(a) = self.lookup_token(pre)
                    && let Some(b) = self.lookup_token(post)
                /*
                && a < token
                && b < token

                 */
                {
                    pairs.insert((a, b), token);
                }
            }
        }

        PairTokenMapVocab::<T>::init(byte_table, pairs).unwrap()
    }
}

impl<T: TokenType> TokenVocabIndex<T> for SpanTokenVocab<T> {
    fn unordered_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.span_map.values().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::{ByteTokenTable, TokenVocabIndex};

    #[test]
    fn test_tokens_iter() {
        type T = u32;

        let byte_table: ByteTokenTable<T> = Default::default();

        let vocab = SpanTokenVocab::<T>::default();

        assert_eq!(vocab.max_token(), byte_table.max_token());
        assert_eq!(&vocab.sorted_tokens(), &byte_table.sorted_tokens());

        let mut span_map = vocab.span_map().clone();

        span_map.insert("apple".as_bytes().to_vec(), 300);
        span_map.insert("banana".as_bytes().to_vec(), 301);
        span_map.insert("pear".as_bytes().to_vec(), 302);

        let vocab = SpanTokenVocab::from(span_map);

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

    #[test]
    fn test_lookup_token() {
        type T = u32;

        let mut span_map: AHashMap<Vec<u8>, T> = Default::default();
        span_map.insert("apple".as_bytes().to_vec(), 300);
        span_map.insert("a".as_bytes().to_vec(), 301);

        let vocab = SpanTokenVocab::<T>::from_span_map(span_map);

        assert_eq!(vocab.lookup_token(b"apple"), Some(300));
        assert_eq!(vocab.lookup_token(b"a"), Some(301));
        assert_eq!(vocab.lookup_token(b"b"), Some('b' as u32));
    }

    #[test]
    fn test_build_pair_vocab() {
        type T = u32;

        let mut span_map: AHashMap<Vec<u8>, T> = Default::default();
        span_map.insert("at".as_bytes().to_vec(), 300);
        span_map.insert("ate".as_bytes().to_vec(), 301);
        span_map.insert("cat".as_bytes().to_vec(), 302);

        let vocab = SpanTokenVocab::from(span_map);

        let pair_vocab = vocab.to_pair_vocab();
        assert_eq!(
            pair_vocab.pairs(),
            &[
                (('a' as u32, 't' as u32), 300),
                ((300, 'e' as u32), 301),
                (('c' as u32, 300), 302)
            ]
            .iter()
            .map(|&(a, b)| (a, b))
            .collect::<AHashMap<_, _>>()
        );
    }
}
