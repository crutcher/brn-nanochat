//! # Text Splitting

use ahash::AHashMap;
use num_traits::Num;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::AddAssign;

/// Split text into words and count occurrences using a regular expression.
pub fn word_counts_from_text<S, K, C>(
    regex: &fancy_regex::Regex,
    text: S,
) -> anyhow::Result<AHashMap<K, C>>
where
    S: AsRef<str>,
    K: for<'a> From<&'a str> + Hash + Eq + Debug,
    C: Num + AddAssign + Default,
{
    let mut m: AHashMap<K, C> = Default::default();
    update_word_counts_from_text(&mut m, regex, text)?;
    Ok(m)
}

/// Update word counts in-place from text using a regular expression.
pub fn update_word_counts_from_text<S, K, C>(
    word_counts: &mut AHashMap<K, C>,
    regex: &fancy_regex::Regex,
    text: S,
) -> anyhow::Result<()>
where
    S: AsRef<str>,
    K: for<'a> From<&'a str> + Hash + Eq + Debug,
    C: Num + AddAssign + Default,
{
    for mat in regex.find_iter(text.as_ref()) {
        let piece = mat?.as_str();
        let k: K = piece.into();
        *word_counts.entry(k).or_default() += C::one();
    }
    Ok(())
}

/// Update word counts inplace from another map.
pub fn update_word_counts<K, C>(
    word_counts: &mut AHashMap<K, C>,
    source: AHashMap<K, C>,
) where
    K: for<'a> From<&'a str> + Hash + Eq + Debug,
    C: Num + AddAssign + Default,
{
    for (k, v) in source {
        *word_counts.entry(k).or_default() += v;
    }
}

/// Options for [`WordCounter`].
#[derive(Debug, Clone)]
pub struct WordCounterOptions {
    /// The regex pattern used for text splitting.
    pub pattern: String,

    /// Whether to use parallel processing for word counting.
    ///
    /// Only applicable if the word counter is created with parallel processing enabled.
    pub parallel: bool,
}

impl Default for WordCounterOptions {
    fn default() -> Self {
        Self {
            pattern: String::from(crate::GPT4_PATTERN),
            parallel: crate::DEFAULT_PARALLEL,
        }
    }
}

impl WordCounterOptions {
    /// Set the parallel processing option.
    pub fn with_parallel(
        self,
        parallel: bool,
    ) -> Self {
        Self { parallel, ..self }
    }

    /// Set the regex pattern used for text splitting.
    pub fn with_pattern(
        self,
        pattern: impl Into<String>,
    ) -> Self {
        Self {
            pattern: pattern.into(),
            ..self
        }
    }
}

/// Word counter structure.
#[derive(Debug)]
pub struct WordCounter<K, C>
where
    K: for<'a> From<&'a str> + Hash + Eq + Debug,
    C: Num + AddAssign + Default + Send,
{
    /// Whether to use parallel processing for word counting.
    parallel: bool,

    /// The regex pattern used for text splitting.
    pattern: String,

    /// The compiled regex pattern.
    regex: fancy_regex::Regex,

    /// The word counts.
    word_counts: AHashMap<K, C>,
}

impl<K, C> WordCounter<K, C>
where
    K: for<'a> From<&'a str> + Hash + Eq + Debug + Send,
    C: Num + AddAssign + Default + Send,
{
    /// Create a new word counter.
    pub fn new(options: WordCounterOptions) -> Self {
        let pattern = options.pattern;
        let regex = fancy_regex::Regex::new(&pattern).unwrap();

        let parallel = options.parallel;

        #[cfg(not(feature = "rayon"))]
        if parallel {
            panic!("Parallel processing requires the `rayon` feature to be enabled.");
        }

        Self {
            parallel,
            pattern,
            regex,
            word_counts: Default::default(),
        }
    }

    /// Get the parallel processing flag.
    pub fn parallel(&self) -> bool {
        self.parallel
    }

    /// Get the regex pattern used for text splitting.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Get the compiled regex pattern.
    pub fn regex(&self) -> &fancy_regex::Regex {
        &self.regex
    }

    /// Get the word counts.
    pub fn word_counts(&self) -> &AHashMap<K, C> {
        &self.word_counts
    }

    /// Release the word counts and return them.
    pub fn release(self) -> AHashMap<K, C> {
        self.word_counts
    }

    /// Update word counts inplace from text.
    pub fn update_from_text<S: AsRef<str>>(
        &mut self,
        text: S,
    ) {
        update_word_counts_from_text(&mut self.word_counts, &self.regex, text).unwrap();
    }

    /// Update word counts inplace from a sample iterator.
    pub fn update_from_samples<S, I>(
        &mut self,
        samples: I,
    ) where
        S: AsRef<str> + Send,
        I: Iterator<Item = S> + Send,
    {
        if self.parallel {
            #[cfg(not(feature = "rayon"))]
            panic!("Parallel processing requires the `rayon` feature to be enabled.");

            #[cfg(feature = "rayon")]
            self.update_from_samples_rayon(samples)
        } else {
            self.update_from_samples_serial(samples);
        }
    }

    /// Update word counts inplace from a sample iterator.
    ///
    /// Uses serial processing, ignoring the `parallel` flag.
    pub fn update_from_samples_serial<S, I>(
        &mut self,
        samples: I,
    ) where
        S: AsRef<str> + Send,
        I: Iterator<Item = S>,
    {
        for sample in samples {
            self.update_from_text(sample);
        }
    }

    /// Update word counts inplace from a sample iterator.
    ///
    /// Uses parallel processing, ignoring the `parallel` flag.
    #[cfg(feature = "rayon")]
    pub fn update_from_samples_rayon<S, I>(
        &mut self,
        samples: I,
    ) where
        S: AsRef<str> + Send,
        I: Iterator<Item = S> + Send,
    {
        use rayon::iter::ParallelBridge;
        use rayon::prelude::*;

        let regex = self.regex.clone();

        let updates: AHashMap<K, C> = samples
            .par_bridge()
            .map(|sample| word_counts_from_text(&regex, sample.as_ref()).unwrap())
            .reduce(
                || AHashMap::new(),
                |mut a, b| {
                    update_word_counts(&mut a, b);
                    a
                },
            );

        self.update_from_word_counts(updates)
    }

    /// Update word counts inplace from a map.
    pub fn update_from_word_counts(
        &mut self,
        word_counts: AHashMap<K, C>,
    ) {
        update_word_counts(&mut self.word_counts, word_counts);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use compact_str::CompactString;
    use num_traits::FromPrimitive;

    const PATTERN: &str = r"\w+";

    fn get_regex() -> fancy_regex::Regex {
        fancy_regex::Regex::new(PATTERN).unwrap()
    }

    #[test]
    fn test_text_to_word_counts() {
        let regex = get_regex();

        let text = "Hello, world! Foo world bar world.";
        let counts: AHashMap<String, u32> = word_counts_from_text(&regex, text).unwrap();
        check_common_counts(counts);
    }

    #[test]
    fn test_update_word_counts() {
        let regex = get_regex();

        let mut counts1: AHashMap<CompactString, usize> =
            word_counts_from_text(&regex, "Hello, world!").unwrap();
        let counts2 = word_counts_from_text(&regex, "Foo world bar world.").unwrap();

        update_word_counts(&mut counts1, counts2);
        check_common_counts(counts1);
    }

    #[test]
    fn test_word_counter() {
        let mut wc: WordCounter<String, u64> =
            WordCounter::new(WordCounterOptions::default().with_pattern(PATTERN));

        let samples = vec!["Hello world", "Foo world bar world"];
        wc.update_from_samples(samples.iter());

        let counts = wc.release();
        check_common_counts(counts);
    }

    fn check_common_counts<K, C>(counts: AHashMap<K, C>)
    where
        K: for<'a> From<&'a str> + Hash + Eq + Ord + Debug,
        C: Num + FromPrimitive + Default + Ord + Debug,
    {
        let mut counts: Vec<(K, C)> = counts.into_iter().collect::<Vec<_>>();
        counts.sort();
        assert_eq!(
            counts,
            vec![
                ("Foo".into(), C::from_usize(1).unwrap()),
                ("Hello".into(), C::from_usize(1).unwrap()),
                ("bar".into(), C::from_usize(1).unwrap()),
                ("world".into(), C::from_usize(3).unwrap()),
            ]
        );
    }
}
