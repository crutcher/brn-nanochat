//! # Text Segmentor

use crate::regex::RegexSupplierHandle;
use crate::regex::alt_list::fixed_alternative_list_regex_wrapper;
use crate::regex::{RegexWrapperPattern, maybe_parallel_regex_supplier};
use crate::vocab::public::size_hints::EXPECTED_BYTES_PER_TOKEN;
use core::ops::Range;

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
    word_re_supplier: RegexSupplierHandle,
    special_re_supplier: Option<RegexSupplierHandle>,
}

impl TextSegmentor {
    /// Create a new text segmentor with the given regex pattern and special words.
    pub fn create<P, S: AsRef<str>>(
        word_pattern: P,
        specials: Option<&[S]>,
    ) -> Self
    where
        P: Into<RegexWrapperPattern>,
    {
        let word_re_supplier = maybe_parallel_regex_supplier(word_pattern.into());

        let special_re_supplier: Option<RegexSupplierHandle> = match specials.as_ref() {
            Some(specials) if !specials.is_empty() => Some(maybe_parallel_regex_supplier(
                fixed_alternative_list_regex_wrapper(specials),
            )),
            _ => None,
        };

        Self::new(word_re_supplier, special_re_supplier)
    }

    /// Create a new text segmentor with the given regex suppliers.
    pub fn new(
        word_re_supplier: RegexSupplierHandle,
        special_re_supplier: Option<RegexSupplierHandle>,
    ) -> Self {
        Self {
            word_re_supplier,
            special_re_supplier,
        }
    }

    /// Find the next special span in the text.
    ///
    /// # Returns
    /// * `Some(Range<usize>)` if a special span is found,
    /// * `None` otherwise.
    pub fn next_special_span<S: AsRef<str>>(
        &self,
        text: S,
    ) -> Option<Range<usize>> {
        self.special_re_supplier
            .as_ref()
            .and_then(|p| p.get_regex().find_iter(text.as_ref()).next())
            .map(|m| m.range())
    }

    /// Split a chunk of text into [`WordRef::Normal`].
    ///
    /// Append to the `words` buffer.
    fn split_append_normal_words<'a>(
        &self,
        text: &'a str,
        words: &mut Vec<WordRef<'a>>,
    ) {
        words.extend(
            self.word_re_supplier
                .get_regex()
                .find_iter(text)
                .map(|m| WordRef::Normal(m.as_str())),
        )
    }

    /// Split a chunk of text into `Vec<WordRef>`.
    ///
    /// Append to the `words` buffer.
    pub fn split_append_words<'a>(
        &self,
        text: &'a str,
        words: &mut Vec<WordRef<'a>>,
    ) {
        let mut current = text;

        while let Some(range) = self.next_special_span(current) {
            let pre = &current[..range.start];
            self.split_append_normal_words(pre, words);

            words.push(WordRef::Special(&current[range.clone()]));

            current = &current[range.end..];
        }

        if !current.is_empty() {
            self.split_append_normal_words(current, words);
        }
    }

    /// Split a chunk of text into `Vec<WordRef>`.
    ///
    /// # Returns
    /// A `Vec<WordRef>` containing the `WordRef`s to `text`..
    pub fn split_words<'a>(
        &self,
        text: &'a str,
    ) -> Vec<WordRef<'a>> {
        let capacity = text.len() as f64 / (EXPECTED_BYTES_PER_TOKEN * 0.5);
        let mut words = Vec::with_capacity(capacity as usize);

        self.split_append_words(text, &mut words);
        words
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;

    #[test]
    fn test_split_words() {
        let segmentor = TextSegmentor::create(
            OA_GPT3_CL100K_WORD_PATTERN,
            Some(&["<|FNORD|>", "<|NORP|>"]),
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
