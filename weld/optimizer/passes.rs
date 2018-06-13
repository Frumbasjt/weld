//! Manages passes over the AST.
//!
//! A pass is a named collection of related transforms applied in sequence until fix point.
use ast::*;
use error::*;

use super::transforms::loop_fusion;
use super::transforms::loop_fusion_2;
use super::transforms::inliner;
use super::transforms::size_inference;
use super::transforms::short_circuit;
use super::transforms::annotator;
use super::transforms::vectorizer;
use super::transforms::unroller;
use super::transforms::adaptive_bloomfilter;
use super::transforms::adaptive_filter_map;
use super::transforms::adaptive_predication;

use std::collections::HashMap;

pub type PassFn = fn(&mut Expr);

/// A single IR to IR transformation.
#[derive(Clone)]
pub struct Transformation {
    pub func: PassFn,
    pub experimental: bool,
    pub adaptive: bool,
}

impl Transformation {
    pub fn new(func: PassFn) -> Transformation {
        Transformation {
            func: func,
            experimental: false,
            adaptive: false,
        }
    }

    pub fn new_experimental(func: PassFn) -> Transformation {
        Transformation {
            func: func,
            experimental: true,
            adaptive: false,
        }
    }

    pub fn new_adaptive(func: PassFn) -> Transformation {
        Transformation {
            func: func,
            experimental: false,
            adaptive: true,
        }
    }

    pub fn new_experiment_adaptive(func: PassFn) -> Transformation {
        Transformation {
            func: func,
            experimental: true,
            adaptive: true,
        }
    }
}

#[derive(Clone)]
pub struct Pass {
    transforms: Vec<Transformation>,
    pass_name: String,
}

impl Pass {
    pub fn new(transforms: Vec<Transformation>, pass_name: &'static str) -> Pass {
        Pass {
            transforms: transforms,
            pass_name: String::from(pass_name),
        }
    }

    pub fn transform(&self, mut expr: &mut Expr, use_experimental: bool, use_adaptive: bool) -> WeldResult<bool> {
        let mut changed = false;
        let mut continue_pass = true;
        let mut before = expr.hash_ignoring_symbols()?;
        while continue_pass {
            println!("{}", self.pass_name);
            for transform in self.transforms.iter() {
                // Skip experimental transformations unless the flag is explicitly set.
                if transform.experimental && !use_experimental {
                    continue;
                }
                // Skip micro-adaptive transformations unless the flag is explicitly set.
                if transform.adaptive && !use_adaptive {
                    continue;
                }
                (transform.func)(&mut expr);
            }
            let after = expr.hash_ignoring_symbols()?;
            continue_pass = !(before == after);
            changed |= continue_pass;
            before = after;
        }
        Ok(changed)
    }

    pub fn pass_name(&self) -> String {
        self.pass_name.clone()
    }
}

lazy_static! {
    pub static ref OPTIMIZATION_PASSES: HashMap<&'static str, Pass> = {
        let mut m = HashMap::new();
        m.insert("inline-apply",
                 Pass::new(vec![Transformation::new(inliner::inline_apply)], "inline-apply"));
        m.insert("inline-let",
                 Pass::new(vec![Transformation::new(inliner::inline_let)], "inline-let"));
        m.insert("inline-zip",
                 Pass::new(vec![Transformation::new(inliner::inline_zips)], "inline-zip"));
        m.insert("loop-fusion",
                 Pass::new(vec![Transformation::new(loop_fusion::fuse_loops_vertical),
                                Transformation::new(loop_fusion_2::fuse_loops_2),
                                Transformation::new(loop_fusion_2::move_merge_before_let),
                                Transformation::new(inliner::inline_get_field),
                                Transformation::new(inliner::inline_let),
                                Transformation::new_experimental(loop_fusion_2::aggressive_inline_let),
                                Transformation::new_experimental(loop_fusion_2::merge_makestruct_loops)],
                 "loop-fusion"));
        m.insert("unroll-static-loop",
                 Pass::new(vec![Transformation::new(unroller::unroll_static_loop)],
                 "unroll-static-loop"));
        m.insert("infer-size",
                 Pass::new(vec![Transformation::new(size_inference::infer_size)],
                 "infer-size"));
        m.insert("inline-literals",
                 Pass::new(vec![Transformation::new(inliner::inline_negate),
                                Transformation::new(inliner::inline_cast)],
                 "inline-literals"));
        m.insert("unroll-structs",
                 Pass::new(vec![Transformation::new(inliner::unroll_structs)],
                 "unroll-structs"));
        m.insert("short-circuit-booleans",
                 Pass::new(vec![Transformation::new(short_circuit::short_circuit_booleans)],
                 "short-circuit-booleans"));
        m.insert("predicate",
                 Pass::new(vec![Transformation::new(vectorizer::predicate_merge_expr),
                                Transformation::new(vectorizer::predicate_simple_expr)],
                 "predicate"));
        m.insert("vectorize",
                 Pass::new(vec![Transformation::new(vectorizer::vectorize)],
                 "vectorize"));
        m.insert("fix-iterate",
                 Pass::new(vec![Transformation::new(annotator::force_iterate_parallel_fors)],
                 "fix-iterate"));
        m.insert("adapt-reorder-filter-projection",
                 Pass::new(vec![Transformation::new_adaptive(adaptive_filter_map::reorder_filter_projection)],
                 "adapt-reorder-filter-projection"));
        m.insert("adapt-bloomfilter",
                 Pass::new(vec![Transformation::new_adaptive(adaptive_bloomfilter::adaptive_bf_phase_1),
                                Transformation::new_adaptive(adaptive_bloomfilter::adaptive_bf_phase_2)],
                 "adapt-bloomfilter"));
        m.insert("adapt-predicate",
                 Pass::new(vec![Transformation::new_adaptive(adaptive_predication::adaptive_predication)],
                 "adapt-predicate"));
        m
    };
}
