//! # Text Splitting

use ahash::AHashMap;

/// Split text into words and count occurrences using a regular expression.
pub fn text_to_word_counts<S, K, C>(
    text: S,
    regex: &fancy_regex::Regex,
) -> anyhow::Result<AHashMap<K, C>>
where
    S: AsRef<str>,
    K: for<'a> From<&'a str> + std::hash::Hash + Eq + std::fmt::Debug,
    C: num_traits::Num + std::ops::AddAssign + Default,
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
    K: for<'a> From<&'a str> + std::hash::Hash + Eq + std::fmt::Debug,
    C: num_traits::Num + std::ops::AddAssign + Default,
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
    K: for<'a> From<&'a str> + std::hash::Hash + Eq + std::fmt::Debug,
    C: num_traits::Num + std::ops::AddAssign + Default,
{
    for (k, v) in source {
        *word_counts.entry(k).or_default() += v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use compact_str::CompactString;

    #[test]
    fn test_text_to_word_counts() {
        let regex = fancy_regex::Regex::new(r"\w+").unwrap();

        let text = "Hello, world! Foo world bar world.";
        let counts = text_to_word_counts(text, &regex).unwrap();

        let mut counts: Vec<(CompactString, i32)> = counts.into_iter().collect::<Vec<_>>();
        counts.sort();
        assert_eq!(
            counts,
            vec![
                ("Foo".into(), 1),
                ("Hello".into(), 1),
                ("bar".into(), 1),
                ("world".into(), 3),
            ]
        );
    }

    #[test]
    fn test_update_word_counts() {
        let regex = fancy_regex::Regex::new(r"\w+").unwrap();

        let mut counts1 = text_to_word_counts("Hello, world!", &regex).unwrap();
        let counts2 = text_to_word_counts("Foo world bar world.", &regex).unwrap();

        update_word_counts(&mut counts1, counts2);

        let mut counts: Vec<(CompactString, i32)> = counts1.into_iter().collect::<Vec<_>>();
        counts.sort();
        assert_eq!(
            counts,
            vec![
                ("Foo".into(), 1),
                ("Hello".into(), 1),
                ("bar".into(), 1),
                ("world".into(), 3),
            ]
        );
    }
}
