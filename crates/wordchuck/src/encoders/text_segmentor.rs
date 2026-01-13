//! # Text Segmentor

use crate::util::regex::{RegexWrapperPattern, RegexWrapperPool, fixed_alternative_list_regex};
use std::ops::Range;

/// Word Reference for [`TextSegmentor`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum WordRef<'a> {
    /// A normal word reference.
    Normal(&'a str),

    /// A special word reference.
    Special(&'a str),
}

/// Word Split + Special Words Segmentor
#[derive(Clone)]
pub struct TextSegmentor {
    normal_pool: RegexWrapperPool,
    special_pool: Option<RegexWrapperPool>,
}

impl TextSegmentor {
    /// Create a new text segmentor with the given regex pattern and special words.
    pub fn new<S: AsRef<str>>(
        pattern: RegexWrapperPattern,
        specials: Option<Vec<S>>,
    ) -> Self {
        let normal_pool = RegexWrapperPool::new(pattern.compile().unwrap().into());

        let special_pool: Option<RegexWrapperPool> = match specials.as_ref() {
            Some(specials) if !specials.is_empty() => Some(RegexWrapperPool::new(
                fixed_alternative_list_regex(specials)
                    .compile()
                    .unwrap()
                    .into(),
            )),
            _ => None,
        };

        Self {
            normal_pool,
            special_pool,
        }
    }

    /// Find the first special word in the given text.
    pub fn find_special(
        &self,
        text: &str,
    ) -> Option<Range<usize>> {
        self.special_pool
            .as_ref()
            .and_then(|p| p.get().find_iter(text).next())
            .map(|m| m.range())
    }

    fn split_append_normal_words<'a>(
        &self,
        text: &'a str,
        words: &mut Vec<WordRef<'a>>,
    ) {
        words.extend(
            self.normal_pool
                .get()
                .find_iter(text)
                .map(|m| WordRef::Normal(m.as_str())),
        )
    }

    /// Split a text into word references.
    pub fn split_words<'a>(
        &self,
        text: &'a str,
    ) -> Vec<WordRef<'a>> {
        let size_hint = 4;
        let mut words = Vec::with_capacity(text.len() / size_hint);

        let mut current = text;

        while let Some(range) = self.find_special(current) {
            let pre = &current[..range.start];
            let end = range.end;
            self.split_append_normal_words(pre, &mut words);
            words.push(WordRef::Special(&current[range]));
            current = &current[end..];
        }

        if !current.is_empty() {
            self.split_append_normal_words(current, &mut words);
        }

        words
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GPT4_PATTERN;

    #[test]
    fn test_split_words() {
        let segmentor = TextSegmentor::new(
            RegexWrapperPattern::Adaptive(GPT4_PATTERN.to_string()),
            Some(vec!["<|FNORD|>", "<|NORP|>"]),
        );

        let buf = "hello<|FNORD|> wor<|NORP|>ld!";

        assert_eq!(
            &segmentor.split_words(buf),
            &vec![
                WordRef::Normal(&buf[..5]),
                WordRef::Special(&buf[5..14]),
                WordRef::Normal(&buf[14..18]),
                WordRef::Special(&buf[18..26]),
                WordRef::Normal(&buf[26..28]),
                WordRef::Normal(&buf[28..]),
            ]
        );
    }
}
