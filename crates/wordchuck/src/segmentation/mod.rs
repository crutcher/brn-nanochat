//! # Text Segmentation

pub mod segmentation_config;
pub mod text_segmentor;

pub use segmentation_config::SegmentationConfig;
pub use text_segmentor::{SpanRef, TextSegmentor};
