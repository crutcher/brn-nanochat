//! # Word Structures

use crate::Pair;
use crate::token_types::Token;
use core::hash::Hash;

/// A word in a BPE-based tokenizer.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Word<T: Token> {
    tokens: Vec<T>,
}

impl<T: Token, S: AsRef<[T]>> From<S> for Word<T> {
    fn from(tokens: S) -> Self {
        Self::from_tokens(tokens)
    }
}

impl<T: Token> Word<T> {
    /// Create a new word from a list of ids.
    pub fn from_tokens<S>(tokens: S) -> Self
    where
        S: AsRef<[T]>,
    {
        let tokens = tokens.as_ref().to_vec();
        Self { tokens }
    }

    /// Create a new word from a byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let tokens: Vec<T> = bytes.iter().map(|&b| T::from_u8(b).unwrap()).collect();
        Self { tokens }
    }

    /// Create a new word from a string slice.
    pub fn from_string<S: AsRef<str>>(s: S) -> Self {
        Self::from_bytes(s.as_ref().as_bytes())
    }

    /// Get a list of ids that make up this word.
    pub fn tokens(&self) -> &[T] {
        &self.tokens
    }

    /// Get the number of ids in this word.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Get an iterator over pairs of ids in this word.
    pub fn pairs<'a>(&'a self) -> impl Iterator<Item = Pair<T>> + 'a {
        self.tokens.windows(2).map(|w| (w[0], w[1]))
    }

    /// Reduce the capacity of the internal vector to fit its contents.
    pub fn shrink_to_fit(&mut self) {
        self.tokens.shrink_to_fit();
    }

    const INC: i32 = 1;
    const DEC: i32 = -1;

    /// Merge all non-overlapping occurrences of `pair -> replacement`.
    ///
    /// # Arguments
    /// * `pair` - the pair to merge.
    /// * `replacement` - the token to replace `pair` with.
    /// * `on_merge` - a callback function to invoke for each incremental pair delta.
    ///   The function is called with:
    ///   - `pair` - the pair that was merged.
    ///   - `delta` - the pair count delta: `+1` for an added pair, `-1` for a removed pair.
    pub fn merge_pair_cb<F>(
        &mut self,
        pair: Pair<T>,
        replacement: T,
        on_merge: &mut F,
    ) where
        F: FnMut(Pair<T>, i32),
    {
        let (a, b) = pair;
        let n = self.tokens.len();

        if n < 2 {
            // Single-token words have no pairs to merge.
            return;
        }

        let mut new_tokens: Vec<T> = Vec::with_capacity(n);

        let mut i = 0;
        while i < n {
            let current = self.tokens[i];

            if i + 1 < n && pair == (current, self.tokens[i + 1]) {
                // Remove Previous Pair?
                if let Some(&x) = new_tokens.last() {
                    on_merge((x, a), Self::DEC);
                    on_merge((x, replacement), Self::INC);
                }

                // Remove Current Pair.
                on_merge(pair, Self::DEC);

                // Remove Next Pair?
                if i + 2 < n {
                    let y = self.tokens[i + 2];
                    on_merge((b, y), Self::DEC);
                    on_merge((replacement, y), Self::INC);
                };

                new_tokens.push(replacement);

                // Skip 'a' and 'b'.
                i += 2;
            } else {
                new_tokens.push(current);
                i += 1;
            }
        }

        self.tokens = new_tokens;
    }

    /// Merge all non-overlapping occurrences of `pair -> replacement`.
    ///
    /// # Arguments
    /// * `pair` - the pair to merge.
    /// * `replacement` - the token to replace `pair` with.
    ///
    /// # Returns
    /// a delta list of pair count deltas for this word:
    /// * `(Pair, +1)` - for each instance of an added `Pair`.
    /// * `(Pair, -1)` - for each instance of a removed `Pair`.
    pub fn merge_pair(
        &mut self,
        pair: Pair<T>,
        replacement: T,
    ) -> Vec<(Pair<T>, i32)> {
        let mut deltas: Vec<(Pair<T>, i32)> = Vec::with_capacity(6);
        self.merge_pair_cb(pair, replacement, &mut |p, d| deltas.push((p, d)));
        deltas
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_constructor() {
        let word = Word::from_tokens(vec![1, 2, 3]);
        assert_eq!(word.tokens(), &[1, 2, 3]);
        assert_eq!(word.len(), 3);
    }

    #[test]
    fn test_word_from() {
        let word: Word<i32> = vec![1, 2, 3].into();
        assert_eq!(word.tokens(), &[1, 2, 3]);

        let word: Word<i32> = [1, 2, 3].into();
        assert_eq!(word.tokens(), &[1, 2, 3]);

        let word: Word<i32> = (&[1, 2, 3]).into();
        assert_eq!(word.tokens(), &[1, 2, 3]);
    }

    #[test]
    fn test_word_from_str() {
        let word: Word<i32> = Word::from_string("hello");
        assert_eq!(word.tokens(), &[104, 101, 108, 108, 111]);
    }

    #[test]
    fn test_word_pairs() {
        let word = Word::from_tokens(vec![1, 2, 3]);
        assert_eq!(word.pairs().collect::<Vec<_>>(), vec![(1, 2), (2, 3)]);
    }

    #[test]
    fn test_word_merge_pair() {
        let mut word = Word::from_tokens(vec![1, 2, 3, 1, 2, 2, 1]);

        let deltas = word.merge_pair((1, 2), 1);
        assert_eq!(word.tokens(), &[1, 3, 1, 2, 1]);

        assert_eq!(
            deltas,
            vec![
                // first match
                ((1, 2), -1),
                ((2, 3), -1),
                // second match
                ((1, 3), 1),
                ((3, 1), -1),
                ((3, 1), 1),
                // third match
                ((1, 2), -1),
                ((2, 2), -1),
                ((1, 2), 1),
            ]
        );
    }

    #[test]
    fn test_word_merge_pair_cb() {
        let mut word = Word::from_tokens(vec![1, 2, 3, 1, 2, 2, 1]);
        let mut deltas = Vec::new();

        word.merge_pair_cb((1, 2), 1, &mut |p, d| {
            deltas.push((p, d));
        });
        assert_eq!(word.tokens(), &[1, 3, 1, 2, 1]);

        assert_eq!(
            deltas,
            vec![
                // first match
                ((1, 2), -1),
                ((2, 3), -1),
                // second match
                ((1, 3), 1),
                ((3, 1), -1),
                ((3, 1), 1),
                // third match
                ((1, 2), -1),
                ((2, 2), -1),
                ((1, 2), 1),
            ]
        );
    }
}
