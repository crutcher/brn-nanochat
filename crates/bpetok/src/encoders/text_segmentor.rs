//! # Text Segmentor

use crate::util::regex::regex_pool::RegexWrapperPool;
use crate::util::regex::regex_wrapper::RegexWrapperPattern;
use std::ops::Range;

/// Word Reference for [`TextSegmentor`].
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

    specials: Option<Vec<String>>,
}

impl TextSegmentor {
    /// Create a new text segmentor with the given regex pattern and special words.
    pub fn new(
        pattern: RegexWrapperPattern,
        specials: Option<Vec<String>>,
    ) -> Self {
        let normal_pool = RegexWrapperPool::new(pattern.compile().unwrap().into());
        Self {
            normal_pool,
            specials,
        }
    }

    /// Find the first special word in the given text.
    pub fn find_special(
        &self,
        text: &str,
    ) -> Option<Range<usize>> {
        // TODO: speed this up; shared special regex.

        match &self.specials {
            Some(specials) => specials
                .iter()
                .filter_map(|special| {
                    text.find(special)
                        .map(|start| start..(start + special.len()))
                })
                .min_by_key(|range| range.start),

            None => None,
        }
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
        loop {
            if current.is_empty() {
                break;
            }

            if let Some(range) = self.find_special(current) {
                let pre = &current[..range.start];
                let end = range.end;
                self.split_append_normal_words(pre, &mut words);
                words.push(WordRef::Special(&text[range]));
                current = &current[end..];
            } else {
                self.split_append_normal_words(current, &mut words);
                break;
            }
        }
        words
    }
}
