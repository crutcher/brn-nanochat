use derive_new::new;

/// A progress struct that can be used to track the progress of a data loader.
///
/// This is meant to be a dep-less stand-in for `burn::data::datoloader::Progress`
#[derive(new, Clone, Debug)]
pub struct CursorProgress {
    /// The number of items that have been processed.
    pub items_processed: usize,

    /// The total number of items that need to be processed.
    pub items_total: usize,
}

impl CursorProgress {
    /// Create a new [`CursorProgress`] with the given number of items total.
    pub fn from_total(items_total: usize) -> Self {
        Self {
            items_processed: 0,
            items_total,
        }
    }

    /// Create a new [`CursorProgress`] with the given number of items processed and total.
    pub fn from_ratio(
        items_processed: usize,
        items_total: usize,
    ) -> Self {
        Self {
            items_processed,
            items_total,
        }
    }

    pub fn progress(&self) -> f64 {
        self.items_processed as f64 / self.items_total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cursor_progress() {
        let progress = CursorProgress::new(10, 20);
        assert_eq!(progress.items_processed, 10);
        assert_eq!(progress.items_total, 20);
    }

    #[test]
    fn test_cursor_progress_ratio() {
        let progress = CursorProgress::from_ratio(10, 20);
        assert_eq!(progress.items_processed, 10);
        assert_eq!(progress.items_total, 20);
        assert_eq!(progress.progress(), 0.5);
    }

    #[test]
    fn test_cursor_progress_ratio_zero() {
        let progress = CursorProgress::from_ratio(0, 20);
        assert_eq!(progress.items_processed, 0);
        assert_eq!(progress.items_total, 20);
        assert_eq!(progress.progress(), 0.0);
    }
}
