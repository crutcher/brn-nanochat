//! # Slice Parser

use burn::tensor::Slice;

/// Parse a [`Slice`].
///
/// # Arguments
/// - `buf`: the buffer to parse.
///
/// # Returns
/// - a `Result<Slice, String>`.
pub fn parse_slice(buf: &str) -> Result<Slice, String> {
    let mut s = buf.trim();
    let make_error = || format!("Invalid Slice: \"{buf}\"");
    let parse = |v: &str| -> Result<isize, String> { v.parse::<isize>().map_err(|_| make_error()) };

    let mut start: isize = 0;
    let mut end: Option<isize> = None;
    let mut step: isize = 1;

    if s.is_empty() {
        return Err(make_error());
    }

    if let Some((head, tail)) = s.split_once(":") {
        step = parse(tail)?;
        s = head;
    }
    if let Some((start_s, end_s)) = s.split_once("..") {
        if !start_s.is_empty() {
            start = parse(start_s)?;
        }
        if !end_s.is_empty() {
            if let Some(end_s) = end_s.strip_prefix('=') {
                end = Some(parse(end_s)? + 1);
            } else {
                end = Some(parse(end_s)?);
            }
        }
    } else {
        if !s.is_empty() {
            start = parse(s)?;
        }
        end = Some(start + 1);
    }

    Ok(Slice::new(start, end, step))
}

#[cfg(test)]
mod tests {
    use crate::slice_parser::parse_slice;
    use burn::tensor::Slice;

    #[test]
    fn test_parse_slice() {
        assert_eq!(parse_slice("1"), Ok(Slice::new(1, Some(2), 1)));
        assert_eq!(parse_slice(".."), Ok(Slice::new(0, None, 1)));
        assert_eq!(parse_slice("..3"), Ok(Slice::new(0, Some(3), 1)));
        assert_eq!(parse_slice("..=3"), Ok(Slice::new(0, Some(4), 1)));

        assert_eq!(parse_slice("-12..3"), Ok(Slice::new(-12, Some(3), 1)));
        assert_eq!(parse_slice("..:-1"), Ok(Slice::new(0, None, -1)));

        assert_eq!(parse_slice("..=3:-2"), Ok(Slice::new(0, Some(4), -2)));

        assert_eq!(parse_slice("").unwrap_err(), "Invalid Slice: \"\"");
        assert_eq!(parse_slice("a").unwrap_err(), "Invalid Slice: \"a\"");
        assert_eq!(parse_slice("..x").unwrap_err(), "Invalid Slice: \"..x\"");
        assert_eq!(
            parse_slice("a:b:c").unwrap_err(),
            "Invalid Slice: \"a:b:c\""
        );
    }
}
