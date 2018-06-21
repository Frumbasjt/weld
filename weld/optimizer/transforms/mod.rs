//! Common transformations on expressions.

pub mod loop_fusion;
pub mod loop_fusion_2;
pub mod inliner;
pub mod size_inference;
pub mod annotator;
pub mod vectorizer;
pub mod short_circuit;
pub mod unroller;
pub mod adaptive_bloomfilter;
pub mod adaptive_filter_map;
pub mod adaptive_predication;
pub mod adaptive_common;