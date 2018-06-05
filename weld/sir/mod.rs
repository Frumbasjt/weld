//! Sequential IR for Weld programs

use std::fmt;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::collections::hash_map::Entry;

use std::vec;

use super::ast::*;
use super::ast::Type::*;
use super::error::*;
use super::util::SymbolGenerator;

extern crate fnv;

pub mod optimizations;

// TODO: make these wrapper types so that you can't pass in the wrong value by mistake
pub type BasicBlockId = usize;
pub type FunctionId = usize;

/// A non-terminating statement inside a basic block.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum StatementKind {
    Assign(Symbol),
    AssignLiteral(LiteralKind),
    BinOp {
        op: BinOpKind,
        left: Symbol,
        right: Symbol,
    },
    Broadcast(Symbol),
    Cast(Symbol, Type),
    CUDF {
        symbol_name: String,
        args: Vec<Symbol>,
    },
    // Compiles to a runtime call that registers a defered let statement
    DeferedAssign{
        id: i32,
        cond_func: FunctionId,
        build_func: FunctionId,
        depends_on: Vec<Symbol>
    },
    // Compiles to a runtime call that gets the result of a defered let
    GetDefered(i32),
    GetField {
        value: Symbol,
        index: u32,
    },
    KeyExists {
        child: Symbol,
        key: Symbol,
    },
    Length(Symbol),
    Lookup {
        child: Symbol,
        index: Symbol,
    },
    MakeStruct(Vec<Symbol>),
    MakeVector(Vec<Symbol>),
    Merge { builder: Symbol, value: Symbol },
    Negate(Symbol),
    NewBuilder {
        args: Vec<Symbol>,
        ty: Type,
    },
    Res(Symbol),
    Select {
        cond: Symbol,
        on_true: Symbol,
        on_false: Symbol,
    },
    Slice {
        child: Symbol,
        index: Symbol,
        size: Symbol,
    },
    StrSlice {
        child: Symbol,
        offset: Symbol,
    },
    Sort {
        child: Symbol,
        keyfunc: SirFunction,
    },
    Serialize(Symbol),
    Deserialize(Symbol),
    ToVec(Symbol),
    Keys(Symbol),
    UnaryOp {
        op: UnaryOpKind,
        child: Symbol,
    },
    BloomFilterContains {
        bf: Symbol,
        item: Symbol
    }
}

impl StatementKind {
    pub fn children(&self) -> vec::IntoIter<&Symbol> {
        use self::StatementKind::*;
        let mut vars = vec![];
        match *self {
            // push any existing symbols that are used (but not assigned) by the statement
            BinOp {
                ref left,
                ref right,
                ..
            } => {
                vars.push(left);
                vars.push(right);
            }
            UnaryOp {
                ref child,
                ..
            } => {
                vars.push(child);
            }
            Cast(ref child, _) => {
                vars.push(child);
            }
            Negate(ref child) => {
                vars.push(child);
            }
            Broadcast(ref child) => {
                vars.push(child);
            }
            Serialize(ref child) => {
                vars.push(child);
            }
            Deserialize(ref child) => {
                vars.push(child);
            }
            Lookup {
                ref child,
                ref index,
            } => {
                vars.push(child);
                vars.push(index);
            }
            KeyExists { ref child, ref key } => {
                vars.push(child);
                vars.push(key);
            }
            Slice {
                ref child,
                ref index,
                ref size,
            } => {
                vars.push(child);
                vars.push(index);
                vars.push(size);
            }
            StrSlice {
                ref child,
                ref offset,
            } => {
                vars.push(child);
                vars.push(offset);
            }
            Sort {
                ref child,
                ..
            } => {
                vars.push(child);
            }
            Select {
                ref cond,
                ref on_true,
                ref on_false,
            } => {
                vars.push(cond);
                vars.push(on_true);
                vars.push(on_false);
            }
            ToVec(ref child) => {
                vars.push(child);
            }
            Keys(ref child) => {
                vars.push(child);
            }
            Length(ref child) => {
                vars.push(child);
            }
            Assign(ref value) => {
                vars.push(value);
            }
            Merge {
                ref builder,
                ref value,
            } => {
                vars.push(builder);
                vars.push(value);
            }
            Res(ref builder) => vars.push(builder),
            GetField { ref value, .. } => vars.push(value),
            AssignLiteral { .. } => {}
            NewBuilder { ref args, .. } => {
                for arg in args {
                    vars.push(arg);
                }
            }
            MakeStruct(ref elems) => {
                for elem in elems {
                    vars.push(elem);
                }
            }
            MakeVector(ref elems) => {
                for elem in elems {
                    vars.push(elem);
                }
            }
            CUDF {
                ref args,
                ..
            } => {
                for arg in args {
                    vars.push(arg);
                }
            }
            BloomFilterContains {
                ref bf,
                ref item
            } => {
                vars.push(bf);
                vars.push(item);
            }
            DeferedAssign { .. } => {
                // Only uses globals which aren't relevant
            }
            GetDefered { .. } => {}
        }
        vars.into_iter()
    }
}

/// A single statement in the SIR, with a RHS statement kind and an optional LHS output symbol.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Statement {
    pub output: Option<Symbol>,
    pub kind: StatementKind,
}

impl Statement {
    pub fn new(output: Option<Symbol>, kind: StatementKind) -> Statement {
        Statement {
            output: output,
            kind: kind,
        }
    }
}

/// Wrapper type to add statements into a program. This object prevents statements from being
/// produced more than once.

/// A site in the program, identified via a `FunctionId` and `BasicBlockId`
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ProgramSite(FunctionId, BasicBlockId);

/// Maps generated statements to the symbol representing the output of that statement in a given
/// site.
type SiteSymbolMap = fnv::FnvHashMap<StatementKind, Symbol>;

struct StatementTracker {
    generated: fnv::FnvHashMap<ProgramSite,SiteSymbolMap>,
}

impl StatementTracker {

    pub fn new() -> StatementTracker {
        StatementTracker {
            generated: fnv::FnvHashMap::default(),
        }
    }

    /// Returns a symbol holding the value of the given `StatementKind` in `(func, block)`. If a
    /// symbol representing this statement does not exist, the statement is added to the program
    /// and a new `Symbol` is returned.
    ///
    /// This function should not be used for statements with _named_ parameters (e.g., identifiers,
    /// parameters in a `Lambda`, or names bound using a `Let` statement.)!
    fn symbol_for_statement(&mut self,
                            prog: &mut SirProgram,
                            func: FunctionId,
                            block: BasicBlockId,
                            sym_ty: &Type,
                            kind: StatementKind) -> Symbol {

        use sir::StatementKind::CUDF;

        let site = ProgramSite(func, block);
        let map = self.generated.entry(site).or_insert(fnv::FnvHashMap::default());

        // CUDFs are the only functions that can have side-effects so we always need to give them
        // a new name.
        if let CUDF { .. } = kind {
            let res_sym = prog.add_local(sym_ty, func);
            prog.funcs[func].blocks[block].add_statement(Statement::new(Some(res_sym.clone()), kind));
            return res_sym;
        }

        // Return the symbol to use.
        match map.entry(kind.clone()) {
            Entry::Occupied(ent) => {
                ent.get().clone()
            }
            Entry::Vacant(ent) => {
                let res_sym = prog.add_local(sym_ty, func);
                prog.funcs[func].blocks[block].add_statement(Statement::new(Some(res_sym.clone()), kind));
                ent.insert(res_sym.clone());
                res_sym
            }
        }
    }

    /// Adds a Statement with a named statement.
    fn named_symbol_for_statement(&mut self,
                                  prog: &mut SirProgram,
                                  func: FunctionId,
                                  block: BasicBlockId,
                                  sym_ty: &Type,
                                  kind: StatementKind,
                                  named_sym: Symbol) {

        let site = ProgramSite(func, block);
        let map = self.generated.entry(site).or_insert(fnv::FnvHashMap::default());

        prog.add_local_named(sym_ty, &named_sym, func);
        prog.funcs[func].blocks[block].add_statement(Statement::new(Some(named_sym.clone()), kind.clone()));
        map.insert(kind, named_sym.clone());
    }

    /// Adds a Statement that writes to an existing symbol (used for global variables)
    fn overwrite_symbol_with_statement(&mut self,
                                       prog: &mut SirProgram,
                                       func: FunctionId,
                                       block: BasicBlockId,
                                       kind: StatementKind,
                                       named_sym: Symbol) {

        let site = ProgramSite(func, block);
        let map = self.generated.entry(site).or_insert(fnv::FnvHashMap::default());

        prog.funcs[func].blocks[block].add_statement(Statement::new(Some(named_sym.clone()), kind.clone()));
        map.insert(kind, named_sym.clone());
    }
}

/// Object that tracks information relevant to micro-adaptivity. Tracks whether the current expression is
/// within a switch expression, and maps symbols of defered let expressions to unique i32s.
struct AdaptiveTracker {
    in_switch: bool,
    defered_ids: BTreeMap<Symbol, i32>,
}

impl AdaptiveTracker {
    fn new() -> AdaptiveTracker {
        AdaptiveTracker {
            in_switch: false,
            defered_ids: BTreeMap::new()
        }
    }

    fn id_for_defered_symbol(&mut self, symbol: &Symbol) -> i32 {
        let next_id = self.defered_ids.len();
        *self.defered_ids.entry(symbol.clone()).or_insert(next_id as i32)
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ParallelForIter {
    pub data: Symbol,
    pub start: Option<Symbol>,
    pub end: Option<Symbol>,
    pub stride: Option<Symbol>,
    pub kind: IterKind,
    // NdIter specific fields
    pub strides: Option<Symbol>,
    pub shape: Option<Symbol>,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ParallelForData {
    pub data: Vec<ParallelForIter>,
    pub builder: Symbol,
    pub data_arg: Symbol,
    pub builder_arg: Symbol,
    pub idx_arg: Symbol,
    pub body: FunctionId,
    pub cont: FunctionId,
    pub innermost: bool,
    /// If the for loop is part of a pipeline in a SwitchFor. In this case the runtime is never used.
    pub switched: bool,
    /// If the for loop is the first in one of the pipeline in a SwitchFor. Implies switched is true.
    pub switch_entry: bool,
    /// If `true`, always invoke parallel runtime for the loop.
    pub always_use_runtime: bool,
    pub grain_size: Option<i32>
}

// Data for a single SwitchFor expression.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct SwitchForData {
    pub flavors: Vec<SwitchFlavorData>,
    pub cont: FunctionId,
    pub grain_size: Option<i32>
}

// Data for a single variant in a SwitchFor expression.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct SwitchFlavorData {
    pub for_func: FunctionId,
    pub for_data: ParallelForData,
    pub lb_arg: Symbol,
    pub ub_arg: Symbol,
    pub conditions: Vec<SwitchFlavorCondition>
}

// Data for a conditional flavor.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum SwitchFlavorCondition {
    Instrumented(Vec<(Symbol, Type)>),
    IfInitialized(Vec<i32>),
}

/// A terminating statement inside a basic block.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Terminator {
    Branch {
        cond: Symbol,
        on_true: BasicBlockId,
        on_false: BasicBlockId,
    },
    JumpBlock(BasicBlockId),
    JumpFunction(FunctionId),
    ProgramReturn(Symbol),
    EndFunction,
    FunctionReturn(Symbol),
    DeferedSetResult {   // runtime call that sets the result of a defered let
        id: i32,
        result: Symbol
    },
    ParallelFor(ParallelForData),
    SwitchFor(SwitchForData),
    Crash,
}

impl Terminator {
    /// Returns Symbols that the `Terminator` depends on.
    pub fn children(&self) -> vec::IntoIter<&Symbol> {
        use self::Terminator::*;
        let mut vars = vec![];
        match *self {
            Branch { ref cond, .. } => {
                vars.push(cond);
            }
            ProgramReturn(ref sym) => {
                vars.push(sym);
            }
            ParallelFor(ref data) => {
                vars.push(&data.builder);
                vars.push(&data.data_arg);
                vars.push(&data.builder_arg);
                vars.push(&data.idx_arg);
                for iter in data.data.iter() {
                    vars.push(&iter.data);
                    if let Some(ref sym) = iter.start {
                        vars.push(sym);
                    }
                    if let Some(ref sym) = iter.end {
                        vars.push(sym);
                    }
                    if let Some(ref sym) = iter.stride {
                        vars.push(sym);
                    }
                }
            }
            SwitchFor(ref data) => {
                for flavor in data.flavors.iter() {
                    vars.push(&flavor.lb_arg);
                    vars.push(&flavor.ub_arg);
                }
            }
            DeferedSetResult { ref result, .. } => { vars.push(result); },
            FunctionReturn(ref sym) => { vars.push(sym); },
            // Explicitly mention those that do not contain children in case more terminators are added later
            JumpBlock(_) => {},
            JumpFunction(_) => {},
            EndFunction => {},
            Crash => {}
        };
        vars.into_iter()
    }

    /// Returns function ids that the`Terminator` references
    pub fn functions(&self) -> vec::IntoIter<FunctionId> {
        use self::Terminator::*;
        let mut funcs = vec![];
        match *self {
            ParallelFor(ref pf) => {
                funcs.push(pf.body);
                funcs.push(pf.cont);
            }
            JumpFunction(jump_func) => {
                funcs.push(jump_func);
            }
            SwitchFor(ref sf) => {
                for flavor in sf.flavors.iter() {
                    funcs.push(flavor.for_func);
                }
                funcs.push(sf.cont);
            }
            Branch { .. } => {}
            JumpBlock(_) => {}
            ProgramReturn(_) => {}
            EndFunction => {}
            FunctionReturn(_) => {}
            Crash => {}
            DeferedSetResult { .. } => {}
        }
        funcs.into_iter()
    }
}

/// A basic block inside a SIR program
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct BasicBlock {
    pub id: BasicBlockId,
    pub statements: Vec<Statement>,
    pub terminator: Terminator,
}

#[derive(Clone, PartialEq, Eq, Hash)]
/// Types of functions that can be lazily compiled.
pub enum LazyFunctionType {
    // SwitchFor flavor argument.
    ForFlavor(SwitchFlavorData, usize),
    // The build function for a defered let.
    DeferedBuild
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SirFunction {
    pub id: FunctionId,
    pub params: BTreeMap<Symbol, Type>,
    pub locals: BTreeMap<Symbol, Type>,
    pub blocks: Vec<BasicBlock>,
    // None if not lazily compiled, some function type otherwise.
    pub lazy: Option<LazyFunctionType>,
    // TODO: tracking globals in the function is fairly ridicilous, temporary
    pub globals: BTreeMap<Symbol, Type>,
}

impl SirFunction {
    /// Gets the Type for a Symbol in the function. Symbols may be either local variables or
    /// parameters.
    pub fn symbol_type(&self, sym: &Symbol) -> WeldResult<&Type> {
        self.locals.get(sym).map(|s| Ok(s)).unwrap_or_else(|| {
            self.params.get(sym).map(|s| Ok(s)).unwrap_or_else(|| {
                self.globals.get(sym).map(|s| Ok(s)).unwrap_or_else(|| {
                    compile_err!("Can't find symbol {}#{}", sym.name, sym.id)
                })
            })
        })
    }

    pub fn is_lazy(&self) -> bool {
        self.lazy.is_some()
    }
}

pub struct SirProgram {
    /// funcs[0] is the main function
    pub funcs: Vec<SirFunction>,
    pub ret_ty: Type,
    pub top_params: Vec<Parameter>,
    pub global_vars: BTreeMap<Symbol, Type>,
    sym_gen: SymbolGenerator,
}

impl SirProgram {
    pub fn new(ret_ty: &Type, top_params: &Vec<Parameter>) -> SirProgram {
        let mut prog = SirProgram {
            funcs: vec![],
            ret_ty: ret_ty.clone(),
            top_params: top_params.clone(),
            global_vars: BTreeMap::new(),
            sym_gen: SymbolGenerator::new(),
        };
        // Add the main function.
        prog.add_func();
        prog
    }

    pub fn add_func(&mut self) -> FunctionId {
        let func = SirFunction {
            id: self.funcs.len(),
            params: BTreeMap::new(),
            blocks: vec![],
            locals: BTreeMap::new(),
            globals: BTreeMap::new(),
            lazy: None,
        };
        self.funcs.push(func);
        self.funcs.len() - 1
    }

    /// Add a local variable of the given type and return a symbol for it.
    pub fn add_local(&mut self, ty: &Type, func: FunctionId) -> Symbol {
        let sym = self.sym_gen.new_symbol(format!("fn{}_tmp", func).as_str());
        self.funcs[func].locals.insert(sym.clone(), ty.clone());
        sym
    }

    /// Add a local variable of the given type and name
    pub fn add_local_named(&mut self, ty: &Type, sym: &Symbol, func: FunctionId) {
        self.funcs[func].locals.insert(sym.clone(), ty.clone());
    }

    pub fn add_global_named(&mut self, ty: &Type, sym: &Symbol) {
        self.global_vars.insert(sym.clone(), ty.clone());
    }
}

impl SirFunction {
    /// Add a new basic block and return its block ID.
    pub fn add_block(&mut self) -> BasicBlockId {
        let block = BasicBlock {
            id: self.blocks.len(),
            statements: vec![],
            terminator: Terminator::Crash,
        };
        self.blocks.push(block);
        self.blocks.len() - 1
    }
}

impl BasicBlock {
    pub fn add_statement(&mut self, statement: Statement) {
        self.statements.push(statement);
    }
}

impl fmt::Display for StatementKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::StatementKind::*;
        match *self {
            Assign(ref value) => write!(f, "{}", value),
            AssignLiteral(ref value) => write!(f, "{}", value),
            BinOp {
                ref op,
                ref left,
                ref right
            } => write!(f, "{} {} {}", op, left, right),
            Broadcast(ref child) => write!(f, "broadcast({})", child),
            Serialize(ref child) => write!(f, "serialize({})", child),
            Deserialize(ref child) => write!(f, "deserialize({})", child),
            Cast(ref child, ref ty) => write!(f, "cast({}, {})", child, ty),
            CUDF {
                ref symbol_name,
                ref args,
            } => {
                write!(f,
                       "cudf[{}]{}",
                       symbol_name,
                       join("(", ", ", ")", args.iter().map(|e| format!("{}", e))))
            },
            DeferedAssign {
                id,
                ref cond_func,
                ref build_func,
                ref depends_on,
            } => {
                write!(f, 
                       "@defer({}, F{}, F{}, {})",
                       id,
                       cond_func,
                       build_func, 
                       join("[", ", ", "]", depends_on.iter().map(|e| format!("{}", e))))
            },
            GetDefered (id) => write!(f, "@get_defered({})", id),
            GetField {
                ref value,
                index,
            } => write!(f, "{}.${}", value, index),
            KeyExists {
                ref child,
                ref key,
            } => write!(f, "keyexists({}, {})", child, key),
            Length(ref child) => write!(f, "len({})", child),
            MakeStruct(ref elems) => {
                write!(f,
                       "{}",
                       join("{", ",", "}", elems.iter().map(|e| format!("{}", e))))
            }
            MakeVector(ref elems) => {
                write!(f,
                       "{}",
                       join("[", ", ", "]", elems.iter().map(|e| format!("{}", e))))
            }
            Merge {
                ref builder,
                ref value,
            } => write!(f, "merge({}, {})", builder, value),
            Negate(ref child) => write!(f, "-{}", child),
            NewBuilder {
                ref args,
                ref ty,
            } => {
                let arg_str = join("(", ",", ")", args.iter().map(|e| format!("{}", e)));
                write!(f, "new {}{}", ty, arg_str)
            }
            Lookup {
                ref child,
                ref index,
            } => write!(f, "lookup({}, {})", child, index),
            Res(ref builder) => write!(f, "result({})", builder),
            Select {
                ref cond,
                ref on_true,
                ref on_false,
            } => write!(f, "select({}, {}, {})", cond, on_true, on_false),
            Slice {
                ref child,
                ref index,
                ref size,
            } => write!(f, "slice({}, {}, {})", child, index, size),
            StrSlice {
                ref child,
                ref offset,
            } => write!(f, "strslice({}, {})", child, offset),
            Sort{ ref child, .. } => write!(f, "sort({})", child),
            ToVec(ref child) => write!(f, "toVec({})", child),
            Keys(ref child) => write!(f, "keys({})", child),
            UnaryOp {
                ref op,
                ref child
            } => write!(f, "{}({})", op, child),
            BloomFilterContains {
                ref bf,
                ref item
            } => write!(f, "bfcontains({}, {})", bf, item)
        }
    }
}

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.output {
            Some(ref sym)               => write!(f, "{} = {}", sym, self.kind),
            None                        => write!(f, "{}", self.kind)
        }
    }
}

impl fmt::Display for Terminator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::Terminator::*;
        match *self {
            Branch {
                ref cond,
                ref on_true,
                ref on_false,
            } => write!(f, "branch {} B{} B{}", cond, on_true, on_false),
            ParallelFor(ref pf) => {
                write!(f, "for [")?;
                for iter in &pf.data {
                    write!(f, "{}, ", iter)?;
                }
                write!(f, "] ")?;
                write!(f,
                       "{} {} {} {} F{} F{} {}",
                       pf.builder,
                       pf.builder_arg,
                       pf.idx_arg,
                       pf.data_arg,
                       pf.body,
                       pf.cont,
                       pf.innermost)?;
                Ok(())
            }
            SwitchFor(ref sf) => {
                write!(f, "switchfor [")?;
                for flavor in sf.flavors.iter() {
                    write!(f, "F{}, ", flavor.for_func)?;
                }
                write!(f, "] F{}", sf.cont)?;
                Ok(())
            }
            JumpBlock(block) => write!(f, "jump B{}", block),
            JumpFunction(func) => write!(f, "jump F{}", func),
            ProgramReturn(ref sym) | FunctionReturn(ref sym) => write!(f, "return {}", sym),
            DeferedSetResult { id, ref result} => write!(f, "@defered_set_result({}, {})", id, result),
            EndFunction => write!(f, "end"),
            Crash => write!(f, "crash"),
        }
    }
}

impl fmt::Display for ParallelForIter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {

        let iterkind = match self.kind {
            IterKind::ScalarIter => "iter",
            IterKind::SimdIter => "simditer",
            IterKind::FringeIter => "fringeiter",
            IterKind::NdIter => "nditer",
            IterKind::RangeIter => "rangeiter",
        };

        if self.shape.is_some() {
            /* NdIter. Note: end or stride aren't important here, so skpping those.
             * */
            write!(f,
                   "{}({}, {}, {}, {})",
                   iterkind,
                   self.data,
                   self.start.clone().unwrap(),
                   self.shape.clone().unwrap(),
                   self.strides.clone().unwrap())
        } else if self.start.is_some() {
            write!(f,
                   "{}({}, {}, {}, {})",
                   iterkind,
                   self.data,
                   self.start.clone().unwrap(),
                   self.end.clone().unwrap(),
                   self.stride.clone().unwrap())
        } else if self.kind != IterKind::ScalarIter {
            write!(f, "{}({})", iterkind, self.data)
        } else {
            write!(f, "{}", self.data)
        }
    }
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "B{}:\n", self.id)?;
        for stmt in &self.statements {
            write!(f, "  {}\n", stmt)?;
        }
        write!(f, "  {}\n", self.terminator)?;
        Ok(())
    }
}

impl fmt::Display for SirFunction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "F{}:\n", self.id)?;
        write!(f, "Params:\n")?;
        let params_sorted: BTreeMap<&Symbol, &Type> = self.params.iter().collect();
        for (name, ty) in params_sorted {
            write!(f, "  {}: {}\n", name, ty)?;
        }
        write!(f, "Locals:\n")?;
        let locals_sorted: BTreeMap<&Symbol, &Type> = self.locals.iter().collect();
        for (name, ty) in locals_sorted {
            write!(f, "  {}: {}\n", name, ty)?;
        }
        for block in &self.blocks {
            write!(f, "{}", block)?;
        }
        Ok(())
    }
}

impl fmt::Display for SirProgram {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Globals:\n")?;
        let globals_sorted: BTreeMap<&Symbol, &Type> = self.global_vars.iter().collect();
        for (sym, ty) in globals_sorted {
            write!(f, "  {}: {}\n", sym, ty.to_string())?;
        }
        write!(f, "\n")?;
        for func in &self.funcs {
            write!(f, "{}\n", func)?;
        }
        Ok(())
    }
}

/// Recursive helper function that injects GetDefered statements at the start of functions 
/// that use a specific symbol whose assignment was defered. Only the first function in 
/// any path from declaration to use is affected.
fn sir_inject_get_defered_helper2(prog: &mut SirProgram, 
                                  func_id: FunctionId, 
                                  defered_sym: &Symbol, 
                                  defered_id: i32, 
                                  visited: &mut HashSet<FunctionId>) -> WeldResult<()> {

    if !visited.insert(func_id) {
        return Ok(());
    }

    // Find any statement or terminator that uses the symbol. If so, insert GetDefered
    // at the start of the function and return.
    // let func = &prog.funcs[func_id];
    for block in prog.funcs[func_id].blocks.clone() {
        for statement in block.statements {
            for sym in statement.kind.children() {
                if sym == defered_sym {
                    // let ty = prog.funcs[func_id].params.get(defered_sym).unwrap();
                    let kind = StatementKind::GetDefered(defered_id);
                    prog.funcs[func_id].blocks[0].statements.insert(0, Statement::new(Some(defered_sym.clone()), kind));
                    return Ok(());
                }
            }
        }
        for sym in block.terminator.children() {
            if sym == defered_sym {
                // let ty = prog.funcs[func_id].params.get(defered_sym).unwrap();
                let kind = StatementKind::GetDefered(defered_id);
                prog.funcs[func_id].blocks[0].statements.insert(0, Statement::new(Some(defered_sym.clone()), kind));
                return Ok(());
            }
        }
    }
    // Make a recursive call for other functions referenced by the terminator.
    for block in prog.funcs[func_id].blocks.clone() {
        for function_id in block.terminator.functions() {
            sir_inject_get_defered_helper2(prog, function_id, defered_sym, defered_id, visited)?;
        }
    }

    Ok(())
}

/// Recursive helper function that identifies DeferedAssign statements, and then calls the second
/// helper to inject the GetDeferred statements.
fn sir_inject_get_defered_helper1(prog: &mut SirProgram, 
                                  func_id: FunctionId, 
                                  visited: &mut HashSet<FunctionId>) -> WeldResult<()> {

    if !visited.insert(func_id) {
        return Ok(());
    }

    // Iterate over every statement of every block in the current function, looking for
    // DeferedAssign statements. At the end of every block we do a recursive call.
    // let func = &prog.funcs[func_id];
    for block in prog.funcs[func_id].blocks.clone() {
        for statement in block.statements {
            if let StatementKind::DeferedAssign { id, .. } = statement.kind {
                if let Some(ref sym) = statement.output {
                    // Make sure there are no uses of this symbol in this function,
                    // otherwise current implementation would break.
                    for block in prog.funcs[func_id].blocks.clone() {
                        for statement in block.statements {
                            for sym2 in statement.kind.children() {
                                if sym == sym2 {
                                    return compile_err!("Internal error: cannot use a DeferedAssign for a symbol that is used in the same function");
                                }
                            }
                        }
                    }
                    // Look for uses of the symbol and inject GetDefered statements.
                    let mut helper2_visited = HashSet::new();
                    for block in prog.funcs[func_id].blocks.clone() {
                        for next_func in block.terminator.functions() {
                            sir_inject_get_defered_helper2(prog, next_func, sym, id, &mut helper2_visited)?;
                        }
                    }
                }
            }
        }
        // Make recursive call into functions referenced by terminator.
        for block in prog.funcs[func_id].blocks.clone() {
            for next_func in block.terminator.functions() {
                sir_inject_get_defered_helper1(prog, next_func, visited)?;
            }
        }
    }
    Ok(())
}

/// Finds DeferedAssign statements, and then injects GetDeferred statements at appropriate locations.
/// Right now it only works correctly if a defered symbol is not used in the same function as where
/// it is declared.
fn sir_inject_get_defered(prog: &mut SirProgram) -> WeldResult<()> {
    sir_inject_get_defered_helper1(prog, 0, &mut HashSet::new())
}

/// Recursive helper function for sir_param_correction. `env` contains the symbol to type mappings
/// that have been defined previously in the program. Any symbols that need to be passed in
/// as closure parameters to func_id will be added to `closure` (so that `func_id`'s
/// callers can also add these symbols to their parameters list, if necessary).
/// `visited` contains functions we have already seen on the way down the function call tree,
/// to prevent infinite recursion when there are loops.
fn sir_param_correction_helper(prog: &mut SirProgram,
                               func_id: FunctionId,
                               env: &mut HashMap<Symbol, Type>,
                               closure: &mut HashSet<Symbol>,
                               visited: &mut HashSet<FunctionId>) {
    // this is needed for cases where params are added outside of sir_param_correction and are not
    // based on variable reads in the function (e.g. in the Iterate case);
    // and when there are loops in the call graph (also in the Iterate case)
    for (name, _) in &prog.funcs[func_id].params {
        closure.insert(name.clone());
    }
    if !visited.insert(func_id) {
        return;
    }
    for (name, ty) in &prog.funcs[func_id].params {
        env.insert(name.clone(), ty.clone());
    }
    for (name, ty) in &prog.funcs[func_id].locals {
        env.insert(name.clone(), ty.clone());
    }

    // All symbols are unique, so there is no need to remove stuff from env at any point.
    for block in prog.funcs[func_id].blocks.clone() {
        let mut vars = vec![];
        for statement in &block.statements {
            vars.extend(statement.kind.children().cloned());
        }
        use self::Terminator::*;
        match block.terminator {
            // push any existing symbols that are used by the terminator
            Branch { ref cond, .. } => {
                vars.push(cond.clone());
            }
            ProgramReturn(ref sym) => {
                vars.push(sym.clone());
            }
            ParallelFor(ref pf) => {
                for iter in pf.data.iter() {
                    vars.push(iter.data.clone());
                    if iter.shape.is_some() {
                        vars.push(iter.start.clone().unwrap());
                        vars.push(iter.end.clone().unwrap());
                        vars.push(iter.stride.clone().unwrap());
                        vars.push(iter.shape.clone().unwrap());
                        vars.push(iter.strides.clone().unwrap());
                    } else if iter.start.is_some() {
                        vars.push(iter.start.clone().unwrap());
                        vars.push(iter.end.clone().unwrap());
                        vars.push(iter.stride.clone().unwrap());
                    }
                }
                vars.push(pf.builder.clone());
            }
            JumpBlock(_) => {}
            JumpFunction(_) => {}
            FunctionReturn(ref sym) => { vars.push(sym.clone()) }
            DeferedSetResult { ref result, .. } => { vars.push(result.clone()) }
            EndFunction => {}
            Crash => {}
            SwitchFor { .. } => {}
        }

        for var in &vars {
            if prog.funcs[func_id].locals.get(&var) == None {
                if prog.global_vars.get(&var) == None {
                    prog.funcs[func_id]
                        .params
                        .insert(var.clone(), env.get(&var).unwrap().clone());
                    closure.insert(var.clone());
                }
            }
        }
        let mut inner_closure = HashSet::new();
        for statement in block.statements {
            // make recursive call for other functions referenced by statements
            match statement.kind {
                StatementKind::DeferedAssign { build_func, .. } => {
                    sir_param_correction_helper(prog, build_func, env, &mut inner_closure, visited);
                }
                _ => {}
            }
        }
        match block.terminator {
            // make a recursive call for other functions referenced by the terminator
            ParallelFor(ref pf) => {
                sir_param_correction_helper(prog, pf.body, env, &mut inner_closure, visited);
                sir_param_correction_helper(prog, pf.cont, env, &mut inner_closure, visited);
            }
            JumpFunction(jump_func) => {
                sir_param_correction_helper(prog, jump_func, env, &mut inner_closure, visited);
            }
            SwitchFor(ref sf) => {
                for flavor in sf.flavors.iter() {
                    sir_param_correction_helper(prog, flavor.for_func, env, &mut inner_closure, visited);
                }
                sir_param_correction_helper(prog, sf.cont, env, &mut inner_closure, visited);
            }
            Branch { .. } => {}
            JumpBlock(_) => {}
            ProgramReturn(_) => {}
            EndFunction => {}
            FunctionReturn(_) => {}
            Crash => {}
            DeferedSetResult { .. } => {}
        }
        for var in inner_closure {
            if prog.funcs[func_id].locals.get(&var) == None {
                prog.funcs[func_id]
                    .params
                    .insert(var.clone(), env.get(&var).unwrap().clone());
                closure.insert(var.clone());
            }
        }
    }
}

/// gen_expr may result in the use of symbols across function boundaries,
/// so ast_to_sir calls sir_param_correction to correct function parameters
/// to ensure that such symbols (the closure) are passed in as parameters.
/// Can be safely called multiple times -- only the necessary param corrections
/// will be performed.
fn sir_param_correction(prog: &mut SirProgram) -> WeldResult<()> {
    let mut env = HashMap::new();
    for (name, ty) in prog.global_vars.iter() {
        env.insert(name.clone(), ty.clone());
    }
    let mut closure = HashSet::new();
    let mut visited = HashSet::new();
    sir_param_correction_helper(prog, 0, &mut env, &mut closure, &mut visited);
    let ref func = prog.funcs[0];
    for name in closure {
        if func.params.get(&name) == None {
            compile_err!("Unbound symbol {}#{}", name.name, name.id)?;
        }
    }
    Ok(())
}

fn sir_get_used_syms(prog: &mut SirProgram, 
                     func_id: FunctionId, 
                     used_syms: &mut HashSet<Symbol>, 
                     visited: &mut HashSet<FunctionId>) {

    if !visited.insert(func_id) {
        return;
    }

    for block in prog.funcs[func_id].blocks.clone() {
        for statement in block.statements {
            for sym in statement.kind.children() {
                used_syms.insert(sym.clone());
            }

            // make recursive call for other functions referenced by statements
            match statement.kind {
                StatementKind::DeferedAssign { build_func, cond_func, .. } => {
                    sir_get_used_syms(prog, build_func, used_syms, visited);
                    sir_get_used_syms(prog, cond_func, used_syms, visited);
                }
                _ => {}
            }
        }
        for sym in block.terminator.children() {
            used_syms.insert(sym.clone());
        }

        // make recursive call for other functions referenced by terminator
        for next_func in block.terminator.functions() {
            sir_get_used_syms(prog, next_func, used_syms, visited);
        }
    }
}

fn sir_remove_unused(prog: &mut SirProgram) -> WeldResult<()> {
    loop {
        let mut changed = false;
        let mut used_syms = HashSet::new();
        let mut visited = HashSet::new();
        for name in prog.global_vars.keys() {
            used_syms.insert(name.clone());
        }
        for name in prog.funcs[0].params.keys() {
            used_syms.insert(name.clone());
        }

        sir_get_used_syms(prog, 0, &mut used_syms, &mut visited);

        for func in prog.funcs.iter_mut() {
            for sym in func.params.clone().keys() {
                if !used_syms.contains(sym) {
                    func.params.remove(sym);
                    changed = true;
                }
            }
            for sym in func.locals.clone().keys() {
                if !used_syms.contains(sym) {
                    func.locals.remove(sym);
                    changed = true;
                }
            }
            let mut i = 0;
            for block in func.blocks.clone() {
                let mut new_statements = vec![];
                for statement in block.statements {
                    if let Some(ref sym) = statement.output {
                        if used_syms.contains(sym) {
                            new_statements.push(statement.clone());
                        } else {
                            changed = true;
                        }
                    } else {
                        new_statements.push(statement.clone());
                    }
                }
                func.blocks[i].statements = new_statements;
                i += 1;
            }
        }

        if !changed {
            break;
        }
    }

    Ok(())
}

/// Convert an AST to a SIR program. Symbols must be unique in expr.
pub fn ast_to_sir(expr: &Expr, multithreaded: bool, lazy_compilation: bool) -> WeldResult<SirProgram> {
    if let ExprKind::Lambda { ref params, ref body } = expr.kind {
        let mut tracker = StatementTracker::new();
        let mut adaptive_tracker = AdaptiveTracker::new();
        let mut prog = SirProgram::new(&expr.ty, params);
        prog.sym_gen = SymbolGenerator::from_expression(expr);
        for tp in params {
            prog.funcs[0].params.insert(tp.name.clone(), tp.ty.clone());
        }
        let first_block = prog.funcs[0].add_block();

        // Generate code for the global vars initialization
        let mut prev_func = 0; 
        let mut prev_block = first_block;
        for (name, expr) in expr.annotations.run_vars().iter() {
            prog.add_global_named(&expr.ty, name);

            let (cur_func, cur_block, val_sym) = gen_expr(expr, &mut prog, prev_func, prev_block, &mut tracker, multithreaded, lazy_compilation, &mut adaptive_tracker)?;
            let kind = StatementKind::Assign(val_sym);
            tracker.overwrite_symbol_with_statement(&mut prog, cur_func, cur_block, kind, name.clone());

            prev_func = cur_func;
            prev_block = cur_block;
        }

        let (res_func, res_block, res_sym) = gen_expr(body, &mut prog, 0, first_block, &mut tracker, multithreaded, lazy_compilation, &mut adaptive_tracker)?;
        prog.funcs[res_func].blocks[res_block].terminator = Terminator::ProgramReturn(res_sym);

        // Make sure each function is aware of the global variables 
        // (not the prettiest solution so change in the future)
        for func in prog.funcs.iter_mut() {
            for (name, ty) in prog.global_vars.iter() {
                func.globals.insert(name.clone(), ty.clone());
            }
        }

        sir_param_correction(&mut prog)?;
        // second call is necessary in the case where there are loops in the call graph, since
        // some parameter dependencies may not have been propagated through back edges
        sir_param_correction(&mut prog)?;
        // look for defered assignments and inject getdefered runtime calls where necessary
        sir_inject_get_defered(&mut prog)?;
        // remove unused symbols
        // sir_remove_unused(&mut prog)?;

        Ok(prog)
    } else {
        compile_err!("Expression passed to ast_to_sir was not a Lambda")
    }
}

/// Helper method for gen_expr. Used to process the fields of ParallelForIter, like "start",
/// "shape" etc. Returns None, or the Symbol associated with the field. It also resets values for
/// cur_func, and cur_block.
fn get_iter_sym(opt : &Option<Box<Expr>>,
            prog: &mut SirProgram,
            cur_func: &mut FunctionId,
            cur_block: &mut BasicBlockId,
            tracker: &mut StatementTracker,
            multithreaded: bool,
            lazy_compilation: bool,
            adaptive_tracker: &mut AdaptiveTracker,
            body_func: FunctionId) -> WeldResult<Option<Symbol>> {
    if let &Some(ref opt_expr) = opt {
        let opt_res = gen_expr(&opt_expr, prog, *cur_func, *cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
        /* TODO pari: Originally, in gen_expr cur_func, and cur_block were also being set - but this
        does not seem to have any effect. Could potentially remove this if it wasn't needed? All
        the tests seem to pass fine without it as well.
        */
        *cur_func = opt_res.0;
        *cur_block = opt_res.1;
        prog.funcs[body_func]
            .params
            .insert(opt_res.2.clone(), opt_expr.ty.clone());
        return Ok(Some(opt_res.2));
    } else {
        return Ok(None);
    };
}

/// Generate code to compute the expression `expr` starting at the current tail of `cur_block`,
/// possibly creating new basic blocks and functions in the process. Return the function and
/// basic block that the expression will be ready in, and its symbol therein.
fn gen_expr(expr: &Expr,
            prog: &mut SirProgram,
            cur_func: FunctionId,
            cur_block: BasicBlockId,
            tracker: &mut StatementTracker,
            multithreaded: bool,
            lazy_compilation: bool,
            adaptive_tracker: &mut AdaptiveTracker)
            -> WeldResult<(FunctionId, BasicBlockId, Symbol)> {
    use self::StatementKind::*;
    use self::Terminator::*;

    let mut cur_func = cur_func;
    let mut cur_block = cur_block;
    
    // If there is a count calls annotation, generate the appropriate code
    if let Some(expr) = expr.annotations.count_calls() {
        if let ExprKind::Ident(sym) = expr.kind.clone() {
            if let Scalar(sk) = expr.ty.clone() {
                let one_literal_kind = match sk {
                    ScalarKind::I8  => LiteralKind::I8Literal(1 as i8),
                    ScalarKind::I16 => LiteralKind::I16Literal(1 as i16),
                    ScalarKind::I32 => LiteralKind::I32Literal(1 as i32),
                    ScalarKind::I64 => LiteralKind::I64Literal(1 as i64),
                    ScalarKind::U8  => LiteralKind::U8Literal(1 as u8),
                    ScalarKind::U16 => LiteralKind::U16Literal(1 as u16),
                    ScalarKind::U32 => LiteralKind::U32Literal(1 as u32),
                    ScalarKind::U64 => LiteralKind::U64Literal(1 as u64),
                    ScalarKind::F32 => LiteralKind::F32Literal((1.0 as f32).to_bits()),
                    ScalarKind::F64 => LiteralKind::F64Literal((1.0 as f64).to_bits()),
                    _ => return compile_err!("Global variable in count_calls annotation must be an integer or float scalar type.")
                };
                let global_ident = constructors::ident_expr(sym.clone(), expr.ty.clone())?;
                let one_literal = constructors::literal_expr(one_literal_kind)?;
                let increment = constructors::binop_expr(BinOpKind::Add, global_ident, one_literal)?;
                
                let (cur_func_tmp, cur_block_tmp, res_val) = gen_expr(&increment, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
                let kind = StatementKind::Assign(res_val);
                tracker.overwrite_symbol_with_statement(prog, cur_func_tmp, cur_block_tmp, kind, sym);

                cur_func = cur_func_tmp;
                cur_block = cur_block_tmp;
            } else {
                return compile_err!("Global variable in count_calls annotation must be an integer or float scalar type.");
            }
        } else {
            return compile_err!("Expression in count_calls annotations must be an identity expression.");
        }
    }

    match expr.kind {
        ExprKind::Ident(ref sym) => { 
            // if adaptive_tracker.defered_ids.contains_key(&sym) {
            //     // insert get_defered
            //     let kind = StatementKind::GetDefered(*adaptive_tracker.defered_ids.get(&sym).unwrap());
            //     tracker.overwrite_symbol_with_statement(prog, cur_func, cur_block, kind, sym.clone());
            // }
            Ok((cur_func, cur_block, sym.clone())) 
        },

        ExprKind::Literal(ref lit) => {
            let kind = AssignLiteral(lit.clone());
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Let {
            ref name,
            ref value,
            ref body,
        } => {
            if let Some(ref cond) = expr.annotations.defered_until() {
                let id = adaptive_tracker.id_for_defered_symbol(name);

                // Add callbacks for runtime
                let cond_func = prog.add_func();
                let cond_block = prog.funcs[cond_func].add_block();
                let build_func = prog.add_func();
                let build_block = prog.funcs[build_func].add_block();

                // Find all global variables that may change the condition of the defered let.
                let mut depends_on = vec![];
                cond.traverse(&mut |ref e| {
                    if let ExprKind::Ident(ref sym) = e.kind {
                        depends_on.push(sym.clone());
                    }
                });
                depends_on.sort();
                depends_on.dedup();

                // Generate call to runtime
                let kind = DeferedAssign {
                    id: id,
                    cond_func: cond_func,
                    build_func: build_func,
                    depends_on: depends_on
                };
                tracker.named_symbol_for_statement(prog, cur_func, cur_block, &value.ty, kind, name.clone());

                // Implement condition function
                let (cond_func, cond_block, cond_res) = gen_expr(cond, prog, cond_func, cond_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?; // maybe in_switch should just be a hardcoded false here?
                prog.funcs[cond_func].blocks[cond_block].terminator = FunctionReturn(cond_res);

                // Implement build function
                let (build_func, build_block, build_res) = gen_expr(value, prog, build_func, build_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
                prog.funcs[build_func].blocks[build_block].terminator = DeferedSetResult {
                    id: id,
                    result: build_res
                };
               
                // Generate remaining expression
                let (cur_func, cur_block, res_sym) = gen_expr(body, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
                Ok((cur_func, cur_block, res_sym))
            } else {
                let (cur_func, cur_block, val_sym) = gen_expr(value, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;

                let kind = Assign(val_sym);
                tracker.named_symbol_for_statement(prog, cur_func, cur_block, &value.ty, kind, name.clone());

                let (cur_func, cur_block, res_sym) = gen_expr(body, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
                Ok((cur_func, cur_block, res_sym))
            }
        }

        ExprKind::BinOp {
            kind,
            ref left,
            ref right,
        } => {
            let (cur_func, cur_block, left_sym) = gen_expr(left, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let (cur_func, cur_block, right_sym) = gen_expr(right, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let kind = BinOp {
                op: kind,
                left: left_sym,
                right: right_sym,
            };
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::UnaryOp {
            kind,
            ref value,
        } => {
            let (cur_func, cur_block, value_sym) = gen_expr(value, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let kind = UnaryOp {
                op: kind,
                child: value_sym,
            };
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Negate(ref child_expr) => {
            let (cur_func, cur_block, child_sym) = gen_expr(child_expr, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let kind = Negate(child_sym);
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Broadcast(ref child_expr) => {
            let (cur_func, cur_block, child_sym) = gen_expr(child_expr, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let kind = Broadcast(child_sym);
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Serialize(ref child_expr) => {
            let (cur_func, cur_block, child_sym) = gen_expr(child_expr, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let kind = Serialize(child_sym);
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Deserialize {ref value, .. } => {
            let (cur_func, cur_block, child_sym) = gen_expr(value, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let kind = Deserialize(child_sym);
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Cast {ref child_expr, .. } => {
            let (cur_func, cur_block, child_sym) = gen_expr(child_expr, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let kind = Cast(child_sym, expr.ty.clone());
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Lookup {
            ref data,
            ref index,
        } => {
            let (cur_func, cur_block, data_sym) = gen_expr(data, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let (cur_func, cur_block, index_sym) = gen_expr(index, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;

            let kind = Lookup {
                child: data_sym,
                index: index_sym.clone(),
            };
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::KeyExists { ref data, ref key } => {
            let (cur_func, cur_block, data_sym) = gen_expr(data, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let (cur_func, cur_block, key_sym) = gen_expr(key, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let kind = KeyExists {
                child: data_sym,
                key: key_sym.clone(),
            };
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Slice {
            ref data,
            ref index,
            ref size,
        } => {
            let (cur_func, cur_block, data_sym) = gen_expr(data, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let (cur_func, cur_block, index_sym) = gen_expr(index, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let (cur_func, cur_block, size_sym) = gen_expr(size, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let kind = Slice {
                child: data_sym,
                index: index_sym.clone(),
                size: size_sym.clone(),
            };
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::StrSlice {
            ref data,
            ref offset,
        } => {
            let (cur_func, cur_block, data_sym) = gen_expr(data, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let (cur_func, cur_block, offset_sym) = gen_expr(offset, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let kind = StrSlice {
                child: data_sym,
                offset: offset_sym.clone(),
            };
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Sort {
            ref data,
            ref keyfunc,
        } => {
            if let ExprKind::Lambda {
                       ref params,
                       ref body,
            } = keyfunc.kind {
                let keyfunc_id = prog.add_func();
                let keyblock = prog.funcs[keyfunc_id].add_block();
                let (keyfunc_id, keyblock, key_sym) = gen_expr(body, prog, keyfunc_id, keyblock, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;

                prog.funcs[keyfunc_id].params.insert(params[0].name.clone(), params[0].ty.clone());
                prog.funcs[keyfunc_id].blocks[keyblock].terminator = Terminator::ProgramReturn(key_sym.clone());

                let (cur_func, cur_block, data_sym) = gen_expr(data, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
                let key_function = prog.funcs[keyfunc_id].clone();

                let kind = Sort {
                    child: data_sym,
                    keyfunc: key_function
                };
                let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
                Ok((cur_func, cur_block, res_sym))
            } else {
                compile_err!("Sort key function expected lambda type, instead {:?} provided", keyfunc.ty)
            }
        }
        ExprKind::Select {
            ref cond,
            ref on_true,
            ref on_false,
        } => {
            let (cur_func, cur_block, cond_sym) = gen_expr(cond, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let (cur_func, cur_block, true_sym) = gen_expr(on_true, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let (cur_func, cur_block, false_sym) = gen_expr(on_false, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let kind = Select {
                cond: cond_sym,
                on_true: true_sym.clone(),
                on_false: false_sym.clone(),
            };
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::ToVec { ref child_expr } => {
            let (cur_func, cur_block, child_sym) = gen_expr(child_expr, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let kind = ToVec(child_sym);
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Keys { ref child_expr } => {
            let (cur_func, cur_block, child_sym) = gen_expr(child_expr, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let kind = Keys(child_sym);
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Length { ref data } => {
            let (cur_func, cur_block, child_sym) = gen_expr(data, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let kind = Length(child_sym);
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::If {
            ref cond,
            ref on_true,
            ref on_false,
        } => {
            let (cur_func, cur_block, cond_sym) = gen_expr(cond, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let true_block = prog.funcs[cur_func].add_block();
            let false_block = prog.funcs[cur_func].add_block();
            prog.funcs[cur_func].blocks[cur_block].terminator = Branch {
                cond: cond_sym,
                on_true: true_block,
                on_false: false_block,
            };
            let (true_func, true_block, true_sym) = gen_expr(on_true, prog, cur_func, true_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let (false_func, false_block, false_sym) = gen_expr(on_false, prog, cur_func, false_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let res_sym = prog.add_local(&expr.ty, true_func);
            prog.funcs[true_func].blocks[true_block].add_statement(Statement::new(Some(res_sym.clone()), Assign(true_sym)));
            prog.funcs[false_func].blocks[false_block].add_statement(Statement::new(Some(res_sym.clone()), Assign(false_sym)));

            if true_func != cur_func || false_func != cur_func {
                // TODO we probably want a better for name for this symbol than whatever res_sym is
                prog.add_local_named(&expr.ty, &res_sym, false_func);
                // the part after the if-else block is split out into a separate continuation
                // function so that we don't have to duplicate this code
                let cont_func = prog.add_func();
                let cont_block = prog.funcs[cont_func].add_block();
                prog.funcs[true_func].blocks[true_block].terminator = JumpFunction(cont_func);
                prog.funcs[false_func].blocks[false_block].terminator = JumpFunction(cont_func);
                Ok((cont_func, cont_block, res_sym))
            } else {
                let cont_block = prog.funcs[cur_func].add_block();
                prog.funcs[true_func].blocks[true_block].terminator = JumpBlock(cont_block);
                prog.funcs[false_func].blocks[false_block].terminator = JumpBlock(cont_block);
                Ok((cur_func, cont_block, res_sym))
            }
        }

        ExprKind::Iterate {
            ref initial,
            ref update_func,
        } => {
            // Generate the intial value.
            let (cur_func, cur_block, initial_sym) = gen_expr(initial, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;

            // Pull out the argument name and function body and validate that things type-check.
            let argument_sym;
            let func_body;
            match update_func.kind {
                ExprKind::Lambda { ref params, ref body } if params.len() == 1 => {
                    argument_sym = &params[0].name;
                    func_body = body;
                    if params[0].ty != initial.ty {
                        return compile_err!("Wrong argument type for body of Iterate");
                    }
                    if func_body.ty != Struct(vec![initial.ty.clone(), Scalar(ScalarKind::Bool)]) {
                        return compile_err!("Wrong return type for body of Iterate");
                    }
                    prog.add_local_named(&params[0].ty, argument_sym, cur_func);
                }
                _ => return compile_err!("Argument of Iterate was not a Lambda")
            }

            prog.funcs[cur_func].blocks[cur_block].add_statement(Statement::new(Some(argument_sym.clone()), Assign(initial_sym)));

            // Check whether the function's body contains any parallel loops. If so, we should put the loop body
            // in a new function because we'll need to jump back to it from continuations. If not, we can just
            // make the loop body be another basic block in the current function.
            let parallel_body = contains_parallel_expressions(func_body);
            let body_start_func = if parallel_body {
                let new_func = prog.add_func();
                new_func
            } else {
                cur_func
            };

            let body_start_block = prog.funcs[body_start_func].add_block();

            // Jump to where the body starts
            if parallel_body {
                prog.funcs[cur_func].blocks[cur_block].terminator = JumpFunction(body_start_func);
            } else {
                prog.funcs[cur_func].blocks[cur_block].terminator = JumpBlock(body_start_block);
            }

            // Generate the loop's body, which will work on argument_sym and produce result_sym.
            // The type of result_sym will be {ArgType, bool} and we will repeat the body if the bool is true.
            let (body_end_func, body_end_block, result_sym) =
                gen_expr(func_body, prog, body_start_func, body_start_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;

            // After the body, unpack the {state, bool} struct into symbols argument_sym and continue_sym.
            let continue_sym = prog.add_local(&Scalar(ScalarKind::Bool), body_end_func);
            if parallel_body {
                // this is needed because sir_param_correction does not add variables only used
                // on the LHS of assignments to the params list
                prog.funcs[body_end_func].params.insert(argument_sym.clone(), initial.ty.clone());
            }
            prog.funcs[body_end_func].blocks[body_end_block].add_statement(
                Statement::new(Some(argument_sym.clone()), GetField { value: result_sym.clone(), index: 0 }));
            prog.funcs[body_end_func].blocks[body_end_block].add_statement(
                Statement::new(Some(continue_sym.clone()), GetField { value: result_sym.clone(), index: 1 }));

            // Create two more blocks so we can branch on continue_sym
            let repeat_block = prog.funcs[body_end_func].add_block();
            let finish_block = prog.funcs[body_end_func].add_block();
            prog.funcs[body_end_func].blocks[body_end_block].terminator =
                Branch { cond: continue_sym, on_true: repeat_block, on_false: finish_block };

            // If we had a parallel body, repeat_block must do a JumpFunction to get back to body_start_func;
            // otherwise it can just do a normal JumpBlock since it should be in the same function.
            if parallel_body {
                assert!(body_end_func != body_start_func);
                prog.funcs[body_end_func].blocks[repeat_block].terminator = JumpFunction(body_start_func);
            } else {
                assert!(body_end_func == cur_func && body_start_func == cur_func);
                prog.funcs[body_end_func].blocks[repeat_block].terminator = JumpBlock(body_start_block);
            }

            // In either case, our final value is available in finish_block.
            Ok((body_end_func, finish_block, argument_sym.clone()))
        }

        ExprKind::Merge {
            ref builder,
            ref value,
        } => {
            // This expression doesn't return a symbol, so just add a statement for it directly
            // instead of calling the tracker.
            let (cur_func, cur_block, builder_sym) = gen_expr(builder, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let (cur_func, cur_block, elem_sym) = gen_expr(value, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            prog.funcs[cur_func].blocks[cur_block].add_statement(Statement::new(None, Merge {
                                                                     builder: builder_sym.clone(),
                                                                     value: elem_sym,
                                                                 }));
            Ok((cur_func, cur_block, builder_sym))
        }

        ExprKind::Res { ref builder } => {
            let (cur_func, cur_block, builder_sym) = gen_expr(builder, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let kind = Res(builder_sym);
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::NewBuilder(ref args) => {
            let mut arg_syms = vec![];
            let mut cur_func = cur_func;
            let mut cur_block = cur_block;
            for a in args.iter() {
                let (cur_func_new, cur_block_new, arg_sym) = gen_expr(&a, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
                cur_func = cur_func_new;
                cur_block = cur_block_new;
                arg_syms.push(arg_sym);
            }

            // NewBuilder is special, since they are stateful objects - we can't alias them.
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(Statement::new(Some(res_sym.clone()), NewBuilder {
                                                                     args: arg_syms,
                                                                     ty: expr.ty.clone(),
                                                                 }));
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::MakeStruct { ref elems } => {
            let mut syms = vec![];
            let (mut cur_func, mut cur_block, mut sym) =
                gen_expr(&elems[0], prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            syms.push(sym);
            for elem in elems.iter().skip(1) {
                let r = gen_expr(elem, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
                cur_func = r.0;
                cur_block = r.1;
                sym = r.2;
                syms.push(sym);
            }
            let kind = MakeStruct(syms);
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::MakeVector { ref elems } => {
            let mut syms = vec![];
            let mut cur_func = cur_func;
            let mut cur_block = cur_block;
            for elem in elems.iter() {
                let r = gen_expr(elem, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
                cur_func = r.0;
                cur_block = r.1;
                let sym = r.2;
                syms.push(sym);
            }
            let kind = MakeVector(syms);
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::CUDF {
            ref sym_name,
            ref args,
            ..
        } => {
            let mut syms = vec![];
            let mut cur_func = cur_func;
            let mut cur_block = cur_block;
            for arg in args.iter() {
                let r = gen_expr(arg, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
                cur_func = r.0;
                cur_block = r.1;
                let sym = r.2;
                syms.push(sym);
            }
            let kind = CUDF {
                args: syms,
                symbol_name: sym_name.clone(),
            };
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::GetField { ref expr, index } => {
            let (cur_func, cur_block, struct_sym) = gen_expr(expr, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let field_ty = match expr.ty {
                super::ast::Type::Struct(ref v) => &v[index as usize],
                _ => {
                    compile_err!("Internal error: tried to get field of type {}",
                              &expr.ty)?
                }
            };

            let kind = GetField {
                value: struct_sym,
                index: index,
            };
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &field_ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::For {
            ref iters,
            ref builder,
            ref func,
        } => {
            if let ExprKind::Lambda {
                       ref params,
                       ref body,
                   } = func.kind {

                let (cur_func, cur_block, builder_sym) =
                    gen_expr(builder, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
                let body_func = prog.add_func();
                let body_block = prog.funcs[body_func].add_block();
                prog.add_local_named(&params[0].ty, &params[0].name, body_func);
                prog.add_local_named(&params[1].ty, &params[1].name, body_func);
                prog.add_local_named(&params[2].ty, &params[2].name, body_func);
                prog.funcs[body_func]
                    .params
                    .insert(builder_sym.clone(), builder.ty.clone());
                let mut cur_func = cur_func;
                let mut cur_block = cur_block;
                let mut pf_iters: Vec<ParallelForIter> = Vec::new();
                for iter in iters.iter() {
                    let data_res = gen_expr(&iter.data, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;

                    cur_func = data_res.0;
                    cur_block = data_res.1;
                    prog.funcs[body_func]
                        .params
                        .insert(data_res.2.clone(), iter.data.ty.clone());
                    let start_sym = try!(get_iter_sym(&iter.start, prog, &mut cur_func, &mut cur_block, 
                                                      tracker, multithreaded, lazy_compilation, adaptive_tracker, body_func));
                    let end_sym = try!(get_iter_sym(&iter.end, prog, &mut cur_func, &mut cur_block, 
                                                    tracker, multithreaded, lazy_compilation, adaptive_tracker, body_func));
                    let stride_sym = try!(get_iter_sym(&iter.stride, prog, &mut cur_func, &mut cur_block, 
                                                       tracker, multithreaded, lazy_compilation, adaptive_tracker, body_func));
                    let shape_sym = try!(get_iter_sym(&iter.shape, prog, &mut cur_func, &mut cur_block, 
                                                       tracker, multithreaded, lazy_compilation, adaptive_tracker, body_func));
                    let strides_sym = try!(get_iter_sym(&iter.strides, prog, &mut cur_func, &mut cur_block, 
                                                        tracker, multithreaded, lazy_compilation, adaptive_tracker, body_func));
                    pf_iters.push(ParallelForIter {
                                      data: data_res.2,
                                      start: start_sym,
                                      end: end_sym,
                                      stride: stride_sym,
                                      kind: iter.kind.clone(),
                                      shape: shape_sym,
                                      strides: strides_sym,
                                  });
                }
                let (body_end_func, body_end_block, _) =
                    gen_expr(body, prog, body_func, body_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
                prog.funcs[body_end_func].blocks[body_end_block].terminator = EndFunction;
                let cont_func = prog.add_func();
                let cont_block = prog.funcs[cont_func].add_block();
                let mut is_innermost = true;
                body.traverse(&mut |ref e| if let ExprKind::For { .. } = e.kind {
                                       is_innermost = false;
                                   });
                prog.funcs[cur_func].blocks[cur_block].terminator =
                    ParallelFor(ParallelForData {
                                    data: pf_iters,
                                    builder: builder_sym.clone(),
                                    data_arg: params[2].name.clone(),
                                    builder_arg: params[0].name.clone(),
                                    idx_arg: params[1].name.clone(),
                                    body: body_func,
                                    cont: cont_func,
                                    innermost: is_innermost,
                                    switched: adaptive_tracker.in_switch,
                                    switch_entry: false,
                                    always_use_runtime: if adaptive_tracker.in_switch { false } else { expr.annotations.always_use_runtime() },
                                    grain_size: if adaptive_tracker.in_switch { None } else { expr.annotations.grain_size().clone() }
                                });
                Ok((cont_func, cont_block, builder_sym))
            } else {
                compile_err!("Argument to For was not a Lambda: {}", func.pretty_print())
            }
        }

        ExprKind::SwitchFor { ref fors } => {
            adaptive_tracker.in_switch = true;

            let mut builder_sym = None;
            let mut flavors = vec![];

            let mut flavor_idx = 0;
            for func in fors {
                if let ExprKind::Lambda { ref params, ref body } = func.kind {
                    let mut lazy = false;

                    // Conditions are always pushed, but an empty vector of symbols/ids means it simply is not active.
                    let mut conditions = vec![];
                    let mut syms_tys = vec![];
                    for expr in func.annotations.switch_instrumented().iter() {
                        lazy = false;
                        if let ExprKind::Ident(ref sym) = expr.kind {
                            syms_tys.push((sym.clone(), expr.ty.clone()));
                        } else {
                            return compile_err!("switch_instrumented annotation expects identity expressions only.");
                        }
                    }
                    conditions.push(SwitchFlavorCondition::Instrumented(syms_tys));
                    let mut ids = vec![];
                    for expr in func.annotations.switch_if_initialized().iter() {
                        // If it depends on defered lets, lazy compile the body function
                        lazy |= true;
                        if let ExprKind::Ident(ref sym) = expr.kind {
                            ids.push(adaptive_tracker.id_for_defered_symbol(sym));
                        } else {
                            return compile_err!("switch_if_initialized annotation expects identity expressions only.");
                        }
                    }
                    conditions.push(SwitchFlavorCondition::IfInitialized(ids));

                    // Build the for function
                    let for_func = prog.add_func();
                    let for_block = prog.funcs[for_func].add_block();
                    prog.add_local_named(&params[0].ty, &params[0].name, for_func);
                    prog.add_local_named(&params[1].ty, &params[1].name, for_func);

                    let (end_func, end_block, builder) = gen_expr(body, prog, for_func, for_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
                    prog.funcs[end_func].blocks[end_block].terminator = EndFunction;

                    builder_sym = Some(builder);
                    
                    let mut flavor_data = None;
                    if let ParallelFor(ref mut pf) = prog.funcs[for_func].blocks[for_block].terminator {
                        pf.switch_entry = true;
                        flavor_data = Some(SwitchFlavorData {
                            for_func: for_func.clone(),
                            for_data: pf.clone(),
                            lb_arg: params[0].name.clone(),
                            ub_arg: params[1].name.clone(),
                            conditions: conditions
                        });
                    }
                    if let Some(flavor_data) = flavor_data {
                        if lazy_compilation && lazy {
                            prog.funcs[for_func].lazy = Some(LazyFunctionType::ForFlavor(flavor_data.clone(), flavor_idx));
                        }
                        flavors.push(flavor_data);
                    } else {
                        unreachable!();
                    }
                } else {
                    return compile_err!("Argument to SwitchFor was not Lambda: {}", func.pretty_print());
                }
                flavor_idx += 1;
            }
            let cont_func = prog.add_func();
            let cont_block = prog.funcs[cont_func].add_block();

            prog.funcs[cur_func].blocks[cur_block].terminator =
                SwitchFor(SwitchForData {
                    flavors: flavors,
                    cont: cont_func.clone(),
                    grain_size: expr.annotations.grain_size()
                });

            adaptive_tracker.in_switch = false;
            Ok((cont_func, cont_block, builder_sym.unwrap()))
        }

        ExprKind::BloomFilterContains { ref bf, ref item } => {
            let (cur_func, cur_block, bf_sym) = gen_expr(bf, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;
            let (cur_func, cur_block, item_sym) = gen_expr(item, prog, cur_func, cur_block, tracker, multithreaded, lazy_compilation, adaptive_tracker)?;

            let kind = BloomFilterContains {
                bf: bf_sym,
                item: item_sym,
            };
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);

            Ok((cur_func, cur_block, res_sym))
        }

        _ => compile_err!("Unsupported expression: {}", expr.pretty_print()),
    }
}

/// Return true if an expression contains parallel for operators
fn contains_parallel_expressions(expr: &Expr) -> bool {
    let mut found = false;
    expr.traverse(&mut |ref e| {
        if let ExprKind::For { .. } = e.kind {
            found = true;
        }
    });
    found
}

fn join<T: Iterator<Item = String>>(start: &str, sep: &str, end: &str, strings: T) -> String {
    let mut res = String::new();
    res.push_str(start);
    for (i, s) in strings.enumerate() {
        if i > 0 {
            res.push_str(sep);
        }
        res.push_str(&s);
    }
    res.push_str(end);
    res
}
