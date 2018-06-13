use ast::*;
use ast::ExprKind::*;
use ast::Type::*;
use ast::BuilderKind::*;
use ast::constructors::*;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use util::SymbolGenerator;
use error::WeldResult;
use ast::ScalarKind::*;
use ast::LiteralKind::*;
use annotation::Annotations;

#[cfg(test)]
use tests::typed_expression;

/// An annotation value that tracks data for adaptive keyexists expressions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AdaptiveBloomFilterData {
    pub dict_ident: Box<Expr>,
    pub bf_ident: Box<Expr>,
    pub dict_try_ident: Box<Expr>,
    pub dict_hit_ident: Box<Expr>,
}

impl AdaptiveBloomFilterData {
    pub fn from_dict(dict: Expr, sym_gen: &mut SymbolGenerator) -> WeldResult<AdaptiveBloomFilterData> {
        if let Ident(ref dict_sym) = dict.kind {
            if let Dict(ref kt, _) = dict.ty {
                let bf_sym = sym_gen.new_symbol(&format!("{}_bf", dict_sym.name));
                let dict_try_sym = sym_gen.new_global(&format!("{}_try", dict_sym.name));
                let dict_hit_sym = sym_gen.new_global(&format!("{}_hit", dict_sym.name));
                let bf = ident_expr(bf_sym, BloomFilter(kt.clone())).unwrap();
                let dict_try = ident_expr(dict_try_sym, Scalar(ScalarKind::F64)).unwrap();
                let dict_hit = ident_expr(dict_hit_sym, Scalar(ScalarKind::F64)).unwrap();
                Ok(AdaptiveBloomFilterData {
                    dict_ident: Box::new(dict.clone()),
                    bf_ident: Box::new(bf),
                    dict_try_ident: Box::new(dict_try),
                    dict_hit_ident: Box::new(dict_hit)
                })
            } else {
                compile_err!("Internal error: dict must be of type Dict")
            }
        } else {
            compile_err!("Internal error: dict must be an Ident")
        }
    }

    pub fn dict_sym(&self) -> &Symbol {
        if let Ident(ref sym) = self.dict_ident.kind {
            sym
        } else {
            unreachable!()
        }
    }

    pub fn bf_sym(&self) -> &Symbol {
        if let Ident(ref sym) = self.bf_ident.kind {
            sym
        } else {
            unreachable!()
        }
    }

    pub fn dict_try_sym(&self) -> &Symbol {
        if let Ident(ref sym) = self.dict_try_ident.kind {
            sym
        } else {
            unreachable!()
        }
    }

    pub fn dict_hit_sym(&self) -> &Symbol {
        if let Ident(ref sym) = self.dict_hit_ident.kind {
            sym
        } else {
            unreachable!()
        }
    }
}

/// An annotation value that tracks data for adaptive loops.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AdaptiveLoopData {
    pub bf_data: BTreeMap<Symbol, AdaptiveBloomFilterData>,
}

impl AdaptiveLoopData {
    pub fn new() -> AdaptiveLoopData {
        AdaptiveLoopData {
            bf_data: BTreeMap::new(),
        }
    }

    /// Get the adaptive bloomfilter data given a certain dictionary. If no such data already exists,
    /// a new one is created, using the provided SymbolGenerator.
    pub fn adaptive_bloomfilter(&mut self, dict: Expr, sym_gen: &mut SymbolGenerator) -> WeldResult<AdaptiveBloomFilterData> {
        if let Ident(ref dict_sym) = dict.kind {
            Ok(self.bf_data.entry(dict_sym.clone())
                           .or_insert(AdaptiveBloomFilterData::from_dict(dict.clone(), sym_gen).unwrap()).clone())
        } else {
            compile_err!("Internal error: dict must be an Ident")
        }
    }

    pub fn get_flavors(&self) -> Vec<FlavorData> {
        let mut flavors = vec![];

        // Build instrumented flavor
        let mut dicts = BTreeMap::new();
        for dict in self.bf_data.keys() {
            dicts.insert(dict.clone(), self.bf_data.get(dict).unwrap().clone());
        }
        flavors.push(FlavorData {
            instrumented_dicts: dicts,
            dicts_using_bfs: BTreeMap::new()
        });

        // Find all combinations of bloom filters turned on or off
        for i in 0..u32::pow(2, self.bf_data.len() as u32) {
            let mut dicts_using_bfs = BTreeMap::new();
            // Each bit j in number i represents a dictionary, 1 = use bf, 0 = dont
            let mut j = 0;
            for (dict_sym, bf_data) in self.bf_data.iter() {
                let bit = i & (1 << j) != 0;
                if bit {
                    dicts_using_bfs.insert(dict_sym.clone(), bf_data.clone());
                }
                j += 1;
            }

            flavors.push(FlavorData {
                instrumented_dicts: BTreeMap::new(),
                dicts_using_bfs: dicts_using_bfs
            });
        }

        flavors
    }
}

pub struct FlavorData {
    // Set of dictionaries that are instrumented.
    pub instrumented_dicts: BTreeMap<Symbol, AdaptiveBloomFilterData>,
    // Set of dictionaries that use a bloom filter in this flavor (maps from symbol to ident)
    pub dicts_using_bfs: BTreeMap<Symbol, AdaptiveBloomFilterData>
}

// PASSES

pub fn adaptive_bf_phase_2(expr: &mut Expr) {
    let mut sym_gen = SymbolGenerator::from_expression(&expr);
    let mut run_vars = BTreeMap::new();
    expr.transform_and_continue(&mut |ref mut e| {
        if let SwitchFor { .. } = e.kind {
            return (None, false);
        }

        if let For { .. } = e.kind {
            if let Some(ref adaptive_data) = e.annotations.adaptive_loop() {
                // For each combination insert the relevant bloom filter checks
                let mut for_exprs = vec![];
                let mut for_annotations = vec![];
                for flavor in adaptive_data.get_flavors().iter() {
                    let mut for_expr = e.clone();
                    let mut annotations = Annotations::new();
                    for_expr.annotations.remove_adaptive_loop();
                    for bf_data in flavor.instrumented_dicts.values() {
                        annotations.push_switch_instrumented(*bf_data.dict_try_ident.clone());
                        annotations.push_switch_instrumented(*bf_data.dict_hit_ident.clone());
                    }
                    for bf_data in flavor.dicts_using_bfs.values() {
                        annotations.push_switch_if_initialized(*bf_data.bf_ident.clone());
                    }
                    
                    for_expr.transform_and_continue(&mut |ref mut e| { 
                        if let If { ref cond, ref on_true, ref on_false } = e.kind {
                            if let Some(ref bf_data) = e.annotations.adaptive_bloomfilter() {
                                if let KeyExists { ref key, .. } = cond.kind {
                                    let mut ifke = e.clone();
                                    ifke.annotations.remove_adaptive_bloomfilter();
                                    if flavor.instrumented_dicts.contains_key(bf_data.dict_sym()) {
                                        let mut instr_cond = *cond.clone();
                                        instr_cond.annotations.set_count_calls(bf_data.dict_try_ident.clone());
                                        let mut instr_on_true = *on_true.clone();
                                        instr_on_true.annotations.set_count_calls(bf_data.dict_hit_ident.clone());
                                        let instr_ifke = if_expr(instr_cond, instr_on_true, *on_false.clone()).unwrap();
                                        return (Some(instr_ifke), true);
                                    } else if flavor.dicts_using_bfs.contains_key(bf_data.dict_sym()) {
                                        let bf_check = bloomfiltercontains_expr(*bf_data.bf_ident.clone(), *key.clone()).unwrap();
                                        let ifbf = if_expr(bf_check, ifke, *on_false.clone()).unwrap();
                                        return (Some(ifbf), true);
                                    } else {
                                        return (Some(ifke), true);
                                    }
                                }
                            }
                        }
                        (None, true)
                    });
                    for_exprs.push(for_expr);
                    // for_annotations.push(Annotations::new());
                    for_annotations.push(annotations);
                }

                let mut prev = switchfor_expr(for_exprs, 
                                              for_annotations,
                                              sym_gen.new_symbol("sw_bld"), 
                                              sym_gen.new_symbol("lb_sym"), 
                                              sym_gen.new_symbol("ub_sym")).unwrap();

                let const_02_expr = literal_expr(F64Literal(0.2f64.to_bits())).unwrap();
                let const_00_expr = literal_expr(F64Literal(0.0f64.to_bits())).unwrap();
                let const_false_expr = literal_expr(BoolLiteral(false)).unwrap();
                for bf_data in adaptive_data.bf_data.values() {
                    let bf_result = gen_bloomfilter(&*bf_data.dict_ident, &mut sym_gen).unwrap();
                    prev = let_expr(bf_data.bf_sym().clone(), bf_result, prev).unwrap();

                    let dict_try_expr = ident_expr(bf_data.dict_try_sym().clone(), Scalar(F64)).unwrap();
                    let dict_hit_expr = ident_expr(bf_data.dict_hit_sym().clone(), Scalar(F64)).unwrap();
                    let cond1_expr = binop_expr(BinOpKind::GreaterThan, dict_try_expr.clone(), const_00_expr.clone()).unwrap();
                    let divide_expr = binop_expr(BinOpKind::Divide, dict_hit_expr, dict_try_expr).unwrap();
                    let cond2_expr = binop_expr(BinOpKind::LessThan, divide_expr, const_02_expr.clone()).unwrap();
                    let if_expr = if_expr(cond1_expr, cond2_expr, const_false_expr.clone()).unwrap();
                    prev.annotations.set_defered_until(Box::new(if_expr));

                    run_vars.insert(bf_data.dict_try_sym().clone(), const_00_expr.clone());
                    run_vars.insert(bf_data.dict_hit_sym().clone(), const_00_expr.clone());
                }
                return (Some(prev), false);
            }
        }

        (None, true)
    });

    for (sym, val) in run_vars.iter() {
        expr.annotations.push_run_var(sym.clone(), val.clone());
    }
}

fn gen_bloomfilter(dict_ident: &Expr, sym_gen: &mut SymbolGenerator) -> WeldResult<Expr> {
    if let Ident(ref dict_sym) = dict_ident.kind {
        if let Dict(ref key_ty, _) = dict_ident.ty {
            // Dictionary and dict keys expressions
            let keys_sym = sym_gen.new_symbol(&format!("{}_keys", dict_sym));
            let keys = keys_expr(dict_ident.clone()).unwrap();
            let keys_ident = ident_expr(keys_sym.clone(), keys.ty.clone()).unwrap();
            let keys_length = length_expr(keys_ident.clone()).unwrap();
            
            // Build the for loop
            let iter = Iter::new_simple(keys_ident);
            let builder = newbuilder_expr(BloomBuilder(key_ty.clone()), vec![keys_length]).unwrap();
            let params = vec![
                Parameter::new(sym_gen.new_symbol("b"), builder.ty.clone()),
                Parameter::new(sym_gen.new_symbol("i"), Scalar(I64)),
                Parameter::new(sym_gen.new_symbol("e"), *key_ty.clone())
            ];
            let b_ident = ident_expr(params[0].name.clone(), builder.ty.clone()).unwrap();
            let e_ident = ident_expr(params[2].name.clone(), *key_ty.clone()).unwrap();
            let merge = merge_expr(b_ident, e_ident).unwrap();
            let func = lambda_expr(params, merge).unwrap();
            let par_loop = for_expr(vec![iter], builder, func, false).unwrap();

            // Declare keys before for loop, and return the result of the whole
            let keys_let = let_expr(keys_sym.clone(), keys, par_loop).unwrap();
            let result = result_expr(keys_let).unwrap();

            return Ok(result);
        }
    }
    compile_err!("Internal error: gen_defered_bloomfilter expects a dict ident")
}

pub fn adaptive_bf_phase_1(expr: &mut Expr) {
    let mut sym_gen = SymbolGenerator::from_expression(&expr);
    expr.transform_and_continue(&mut |ref mut e| {
        if let For { .. } = e.kind {
            // Prevent infinite loops (we never need more than one pass)
            if let Some(data) = e.annotations.adaptive_loop() {
                if data.bf_data.len() > 0 {
                    return (None, false);
                }
            }

            // Keep track of all symbols that were declared in this loop.
            let mut declared = BTreeSet::new();
            e.traverse(&mut |ref mut e| {
                if let Let { ref name, .. } = e.kind {
                    declared.insert(name.clone());
                }
            });

            // Mark all if statements with a keyexists expression as bloomfilter
            // candidates, but only if the dictionary was declared outside of the loop.
            let mut adaptive_data = AdaptiveLoopData::new();
            e.transform_and_continue(&mut |ref mut e| {
                if let If { ref cond, .. } = e.kind {
                    if let KeyExists { ref data, .. } = cond.kind {
                        if let Ident(ref symbol) = data.kind {
                            if !declared.contains(symbol) {
                                let adaptive_bloomfilter = adaptive_data.adaptive_bloomfilter(*data.clone(), &mut sym_gen).unwrap();
                                e.annotations.set_adaptive_bloomfilter(adaptive_bloomfilter);
                            }
                        }
                    }
                }
                (None, true)
            });

            // No point in continueing if there no expressions were marked.
            if adaptive_data.bf_data.len() == 0 {
                return (None, false);
            }

            // Annotate the for loop as an adaptive loop and pass it the bloomfilter symbols.
            let mut result = e.clone();
            result.annotations.set_adaptive_loop(adaptive_data);

            return (Some(result), false);
        }
        (None, true)
    });
}