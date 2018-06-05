use std::fmt;
use std::vec;
use std::collections::BTreeMap;

use ast::Expr;
use ast::Symbol;
use optimizer::transforms::adaptive::AdaptiveLoopData;
use optimizer::transforms::adaptive::AdaptiveBloomFilterData;
use ast::PrettyPrint;

/// A kind of annotation that can be set on an expression.
#[derive(Clone, Debug, Ord, PartialOrd, PartialEq, Eq, Hash)]
pub enum AnnotationKind {
    BuilderImplementation,
    Predicate,
    Vectorize,
    TileSize,
    GrainSize,
    AlwaysUseRuntime,
    Size,
    LoopSize,
    BranchSelectivity,
    NumKeys,
    // Internal annotations
    RunVars,
    CountCalls,
    DeferedUntil,
    SwitchInstrumented,
    SwitchIfInitialized,
    AdaptiveBloomFilter,
    AdaptiveLoop
}

impl fmt::Display for AnnotationKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}",
               match *self {
                   AnnotationKind::BuilderImplementation => "impl",
                   AnnotationKind::TileSize => "tile_size",
                   AnnotationKind::GrainSize => "grain_size",
                   AnnotationKind::Size => "size",
                   AnnotationKind::LoopSize => "loopsize",
                   AnnotationKind::BranchSelectivity => "branch_selectivity",
                   AnnotationKind::NumKeys => "num_keys",
                   AnnotationKind::Predicate => "predicate",
                   AnnotationKind::Vectorize => "vectorize",
                   AnnotationKind::AlwaysUseRuntime => "always_use_runtime",
                   AnnotationKind::RunVars => "run_vars",
                   AnnotationKind::CountCalls => "count_calls",
                   AnnotationKind::DeferedUntil => "defered_until",
                   AnnotationKind::SwitchInstrumented => "switch_instrumented",
                   AnnotationKind::SwitchIfInitialized => "switch_if_initialized",
                   AnnotationKind::AdaptiveBloomFilter => "adaptive_bloomfilter",
                   AnnotationKind::AdaptiveLoop => "adaptive_loop"
               })
    }
}

#[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
/// A annotation value for the way a builder is implemented.
pub enum BuilderImplementationKind {
    Local,
    Global,
}

impl fmt::Display for BuilderImplementationKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use annotation::BuilderImplementationKind::*;
        let text = match *self {
            Local => "local",
            Global => "global",
        };
        f.write_str(text)
    }
}

/// An internal representation of annotation values.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AnnotationValue {
    VBuilderImplementation(BuilderImplementationKind),
    VTileSize(i32),
    VGrainSize(i32),
    VSize(i64),
    VLoopSize(i64),
    VNumKeys(i64),
    VBranchSelectivity(i32), // Fractions of 10,000
    VRunVars(BTreeMap<Symbol, Expr>),
    VCountCalls(Box<Expr>),
    VDeferedUntil(Box<Expr>),
    VSwitchInstrumented(Vec<Expr>),
    VSwitchIfInitialized(Vec<Expr>),
    VAdaptiveBloomFilter(AdaptiveBloomFilterData),
    VAdaptiveLoop(AdaptiveLoopData),
    VPredicate,
    VVectorize,
    VAlwaysUseRuntime,
}

impl fmt::Display for AnnotationValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
       write!(f, "{}",
              match *self {
                    AnnotationValue::VBuilderImplementation(ref kind) => format!("{}", kind),
                    AnnotationValue::VTileSize(ref v) => format!("{}", v),
                    AnnotationValue::VGrainSize(ref v) => format!("{}", v),
                    AnnotationValue::VSize(ref v) => format!("{}", v),
                    AnnotationValue::VLoopSize(ref v) => format!("{}", v),
                    AnnotationValue::VBranchSelectivity(ref v) => format!("{}", v),
                    AnnotationValue::VNumKeys(ref v) => format!("{}", v),
                    AnnotationValue::VRunVars(ref v) => {
                        v.iter().map(|(s,e)| format!("{}={}", s, e.pretty_print())).collect::<Vec<_>>().join(",")
                    },
                    AnnotationValue::VCountCalls(ref v) => v.pretty_print(),
                    AnnotationValue::VDeferedUntil(ref expr) => expr.pretty_print(),
                    AnnotationValue::VSwitchInstrumented(ref v) => {
                        v.iter().map(|e| e.pretty_print()).collect::<Vec<_>>().join(",")
                    },
                    AnnotationValue::VSwitchIfInitialized(ref v) => {
                        v.iter().map(|e| e.pretty_print()).collect::<Vec<_>>().join(",")
                    },
                    AnnotationValue::VAdaptiveBloomFilter(ref v) => format!("{}", v.dict_sym()),
                    AnnotationValue::VAdaptiveLoop(_) => "true".to_string(),
                    // These are flags, so their existence indicates that the value is `true`.
                    AnnotationValue::VPredicate => "true".to_string(),
                    AnnotationValue::VVectorize => "true".to_string(),
                    AnnotationValue::VAlwaysUseRuntime => "true".to_string(),
               })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Annotations {
    pub values: BTreeMap<AnnotationKind, AnnotationValue>,
}

impl Annotations {
    pub fn new() -> Annotations {
        return Annotations {
            values: BTreeMap::new(),
        };
    }

    pub fn with_values(values: BTreeMap<AnnotationKind, AnnotationValue>) -> Annotations {
        return Annotations {
            values: values
        };
    }

    pub fn exprs(&self) -> vec::IntoIter<&Expr> {
        let mut result = vec![];
        for av in self.values.values() {
            match *av {
                AnnotationValue::VRunVars(ref exprs) => result.extend(exprs.values()),
                AnnotationValue::VCountCalls(ref expr) => result.push(expr.as_ref()),
                AnnotationValue::VDeferedUntil(ref expr) => result.push(expr.as_ref()),
                AnnotationValue::VSwitchInstrumented(ref exprs) => result.extend(exprs),
                AnnotationValue::VSwitchIfInitialized(ref exprs) => result.extend(exprs),
                _ => {}
            }
        }
        result.into_iter()
    }

    pub fn exprs_mut(&mut self) -> vec::IntoIter<&mut Expr> {
        let mut result = vec![];
        for av in self.values.values_mut() {
            match *av {
                AnnotationValue::VRunVars(ref mut exprs) => result.extend(exprs.values_mut()),
                AnnotationValue::VCountCalls(ref mut expr) => result.push(expr.as_mut()),
                AnnotationValue::VDeferedUntil(ref mut expr) => result.push(expr.as_mut()),
                AnnotationValue::VSwitchInstrumented(ref mut exprs) => result.extend(exprs),
                AnnotationValue::VSwitchIfInitialized(ref mut exprs) => result.extend(exprs),
                _ => {}
            }
        }
        result.into_iter()
    }

    pub fn builder_implementation(&self) -> Option<BuilderImplementationKind> {
        if let Some(s) = self.values.get(&AnnotationKind::BuilderImplementation) {
            if let AnnotationValue::VBuilderImplementation(ref s) = *s {
                return Some(s.clone());
            }
        }
        None
    }

    pub fn set_builder_implementation(&mut self, value: BuilderImplementationKind) {
        self.values.insert(AnnotationKind::BuilderImplementation, AnnotationValue::VBuilderImplementation(value));
    }

    pub fn tile_size(&self) -> Option<i32> {
        if let Some(s) = self.values.get(&AnnotationKind::TileSize) {
            if let AnnotationValue::VTileSize(ref s) = *s {
                return Some(s.clone());
            }
        }
        None
    }

    pub fn set_tile_size(&mut self, value: i32) {
        self.values.insert(AnnotationKind::TileSize, AnnotationValue::VTileSize(value));
    }

    pub fn grain_size(&self) -> Option<i32> {
        if let Some(s) = self.values.get(&AnnotationKind::GrainSize) {
            if let AnnotationValue::VGrainSize(ref s) = *s {
                return Some(s.clone());
            }
        }
        None
    }

    pub fn set_grain_size(&mut self, value: i32) {
        self.values.insert(AnnotationKind::GrainSize, AnnotationValue::VGrainSize(value));
    }

    pub fn size(&self) -> Option<i64> {
        if let Some(s) = self.values.get(&AnnotationKind::Size) {
            if let AnnotationValue::VSize(ref s) = *s {
                return Some(s.clone());
            }
        }
        None
    }

    pub fn set_size(&mut self, value: i64) {
        self.values.insert(AnnotationKind::Size, AnnotationValue::VSize(value));
    }

    pub fn loopsize(&self) -> Option<i64> {
        if let Some(s) = self.values.get(&AnnotationKind::LoopSize) {
            if let AnnotationValue::VLoopSize(ref s) = *s {
                return Some(s.clone());
            }
        }
        None
    }

    pub fn set_loopsize(&mut self, value: i64) {
        self.values.insert(AnnotationKind::LoopSize, AnnotationValue::VLoopSize(value));
    }

    pub fn num_keys(&self) -> Option<i64> {
        if let Some(s) = self.values.get(&AnnotationKind::NumKeys) {
            if let AnnotationValue::VNumKeys(ref s) = *s {
                return Some(s.clone());
            }
        }
        None
    }

    pub fn set_num_keys(&mut self, value: i64) {
        self.values.insert(AnnotationKind::NumKeys, AnnotationValue::VNumKeys(value));
    }

    pub fn branch_selectivity(&self) -> Option<i32> {
        if let Some(s) = self.values.get(&AnnotationKind::BranchSelectivity) {
            if let AnnotationValue::VBranchSelectivity(ref s) = *s {
                return Some(s.clone());
            }
        }
        None
    }

    pub fn set_branch_selectivity(&mut self, value: i32) {
        self.values.insert(AnnotationKind::BranchSelectivity, AnnotationValue::VBranchSelectivity(value));
    }

    pub fn predicate(&self) -> bool {
        return self.values.contains_key(&AnnotationKind::Predicate);
    }

    pub fn set_predicate(&mut self, val: bool) {
        if val {
            self.values.insert(AnnotationKind::Predicate, AnnotationValue::VPredicate);
        } else {
            self.values.remove(&AnnotationKind::Predicate);
        }
    }

    pub fn vectorize(&self) -> bool {
        return self.values.contains_key(&AnnotationKind::Vectorize);
    }

    pub fn set_vectorize(&mut self, val: bool) {
        if val {
            self.values.insert(AnnotationKind::Vectorize, AnnotationValue::VVectorize);
        } else {
            self.values.remove(&AnnotationKind::Vectorize);
        }
    }

    pub fn always_use_runtime(&self) -> bool {
        return self.values.contains_key(&AnnotationKind::AlwaysUseRuntime);
    }

    pub fn set_always_use_runtime(&mut self, val: bool) {
        if val {
            self.values.insert(AnnotationKind::AlwaysUseRuntime, AnnotationValue::VAlwaysUseRuntime);
        } else {
            self.values.remove(&AnnotationKind::AlwaysUseRuntime);
        }
    }

    pub fn run_vars(&self) -> BTreeMap<Symbol, Expr> {
        if let Some(map) = self.values.get(&AnnotationKind::RunVars) {
            if let AnnotationValue::VRunVars(ref map) = *map {
                return map.clone();
            }
        }
        BTreeMap::new()
    }

    pub fn push_run_var(&mut self, sym: Symbol, expr: Expr) {
        let v = self.values
                    .entry(AnnotationKind::RunVars)
                    .or_insert(AnnotationValue::VRunVars(BTreeMap::new()));

        if let AnnotationValue::VRunVars(ref mut vec) = *v {
            vec.insert(sym.clone(), expr);
        }
    }

    pub fn count_calls(&self) -> Option<Box<Expr>> {
        if let Some(v) = self.values.get(&AnnotationKind::CountCalls) {
            if let AnnotationValue::VCountCalls(ref expr) = *v {
                return Some(expr.clone());
            }
        }
        None
    }

    pub fn set_count_calls(&mut self, val: Box<Expr>) {
        self.values.insert(AnnotationKind::CountCalls, AnnotationValue::VCountCalls(val));
    }

    pub fn defered_until(&self) -> Option<(Box<Expr>)> {
        if let Some(v) = self.values.get(&AnnotationKind::DeferedUntil) {
            if let AnnotationValue::VDeferedUntil(ref expr) = *v {
                return Some(expr.clone());
            }
        }
        None
    }

    pub fn set_defered_until(&mut self, expr: Box<Expr>) {
        self.values.insert(AnnotationKind::DeferedUntil, AnnotationValue::VDeferedUntil(expr));
    }

    pub fn remove_defered_until(&mut self) {
        self.values.remove(&AnnotationKind::DeferedUntil);
    }

    pub fn switch_instrumented(&self) -> Vec<Expr> {
        if let Some(v) = self.values.get(&AnnotationKind::SwitchInstrumented) {
            if let AnnotationValue::VSwitchInstrumented(ref vec) = * v {
                return vec.clone();
            }
        }
        vec![]
    }

    pub fn set_switch_instrumented(&mut self, globals: Vec<Expr>) {
        self.values.insert(AnnotationKind::SwitchInstrumented, AnnotationValue::VSwitchInstrumented(globals));
    }

    pub fn push_switch_instrumented(&mut self, global: Expr) {
        let v = self.values
                    .entry(AnnotationKind::SwitchInstrumented)
                    .or_insert(AnnotationValue::VSwitchInstrumented(vec![]));
        
        if let AnnotationValue::VSwitchInstrumented(ref mut vec) = *v {
            vec.push(global);
        }
    }

    pub fn switch_if_initialized(&self) -> Vec<Expr> {
        if let Some(v) = self.values.get(&AnnotationKind::SwitchIfInitialized) {
            if let AnnotationValue::VSwitchIfInitialized(ref ids) = *v {
                return ids.clone();
            }
        }
        vec![]
    }

    pub fn set_switch_if_initialized(&mut self, ids: Vec<Expr>) {
        self.values.insert(AnnotationKind::SwitchIfInitialized, AnnotationValue::VSwitchIfInitialized(ids));
    }

    pub fn push_switch_if_initialized(&mut self, id: Expr) {
        let v = self.values
                    .entry(AnnotationKind::SwitchIfInitialized)
                    .or_insert(AnnotationValue::VSwitchIfInitialized(vec![]));
        
        if let AnnotationValue::VSwitchIfInitialized(ref mut ids) = *v {
            ids.push(id);
        }
    }

    pub fn set_adaptive_bloomfilter(&mut self, val: AdaptiveBloomFilterData) {
        self.values.insert(AnnotationKind::AdaptiveBloomFilter, AnnotationValue::VAdaptiveBloomFilter(val));
    }

    pub fn remove_adaptive_bloomfilter(&mut self) {
        self.values.remove(&AnnotationKind::AdaptiveBloomFilter);
    }

    pub fn adaptive_bloomfilter(&self) -> Option<AdaptiveBloomFilterData> {
        if let Some(v) = self.values.get(&AnnotationKind::AdaptiveBloomFilter) {
            if let AnnotationValue::VAdaptiveBloomFilter(ref data) = *v {
                return Some(data.clone());
            }
        }
        None
    }

    pub fn set_adaptive_loop(&mut self, val: AdaptiveLoopData) {
        self.values.insert(AnnotationKind::AdaptiveLoop, AnnotationValue::VAdaptiveLoop(val));
    }

    pub fn remove_adaptive_loop(&mut self) {
        self.values.remove(&AnnotationKind::AdaptiveLoop);
    }

    pub fn adaptive_loop(&self) -> Option<AdaptiveLoopData> {
        if let Some(v) = self.values.get(&AnnotationKind::AdaptiveLoop) {
            if let AnnotationValue::VAdaptiveLoop(ref data) = *v {
                return Some(data.clone());
            }
        }
        None
    }

    pub fn is_empty(&self) -> bool {
        return self.values.is_empty();
    }
}

impl fmt::Display for Annotations {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut annotations = Vec::new();
        for (ref kind, ref value) in self.values.iter() {
            annotations.push(format!("{}:{}", kind, value));
        }

        // Sort the annotations alphabetically when displaying them so the result is deterministic.
        annotations.sort();

        if annotations.len() == 0 {
            write!(f, "")
        } else {
            write!(f, "@({})", annotations.join(","))
        }
    }
}