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

pub fn reorder_filter_projection(expr: &mut Expr) {
    let mut sym_gen = SymbolGenerator::from_expression(&expr);
    expr.transform_and_continue_res(&mut |ref mut expr| {
        if let SwitchFor { .. } = expr.kind {
            return Ok((None, false));
        }

        if let For { ref iters, builder: ref init_builder, ref func } = expr.kind {
            if let Some ( _ ) = valid_builder(init_builder) {
                if let Lambda { ref params, ref body } = func.kind {
                    if let If { ref cond, ref on_true, ref on_false } = body.kind {
                        if let Merge { ref builder, ref value } = on_true.kind {
                            if let Ident(ref name) = on_false.kind {
                                if let Ident(ref name2) = builder.kind {
                                    if name == name2 {
                                        // Check if the loop maps data
                                        let mut is_proj = false;
                                        value.traverse(&mut |ref e| {
                                            match e.kind {
                                                BinOp { .. } | UnaryOp { .. } => { is_proj = true; },
                                                _ => {}
                                            }
                                        });
                                        if !is_proj {
                                            return Ok((None, true));
                                        }
                                        // Check if loop contains lookup
                                        let mut contains_lookup = false;
                                        value.traverse_and_continue(&mut |ref e| {
                                            match e.kind {
                                                Lookup { .. } => { contains_lookup = true; },
                                                _ => {}
                                            }
                                            !contains_lookup
                                        });
                                        if contains_lookup {
                                            return Ok((None, true));
                                        }
                                        // Check if start/stop/stride
                                        if iters.iter().any(|i| i.start.is_some()) {
                                            return Ok((None, true));
                                        }

                                        let bld_sym = sym_gen.new_symbol("sw_bld");
                                        let lb_sym = sym_gen.new_symbol("lb");
                                        let ub_sym = sym_gen.new_symbol("ub");
                                        let par_syms = params.iter().map(|p| sym_gen.new_symbol(&p.name.name)).collect::<Vec<_>>();

                                        // Check if any of the iters contain a literal. If so, replace with identity and create corresponding let expressions later
                                        let mut data_names = Vec::new();
                                        let iters = iters.clone()
                                            .iter_mut()
                                            .map(|i| {
                                                let mut i = i.clone();
                                                // If the data is a vector literal, replace it with an identity expression, and keep track of the name
                                                // so that we can generate a let expression for it later
                                                if let MakeVector { .. } = (*i.data).kind {
                                                    let data = i.data.clone();
                                                    let data_name = sym_gen.new_symbol("a");
                                                    let data_ty = data.ty.clone();
                                                    i.data = Box::new(ident_expr(data_name.clone(), data_ty).unwrap());
                                                    data_names.push((*data, data_name));
                                                };
                                                i
                                            })
                                            .collect::<Vec<_>>();
                                        
                                        // Create new loop and switch
                                        let map_loop = map_for(&iters, params, *on_true.clone(), lb_sym.clone(), ub_sym.clone(), par_syms).unwrap();
                                        let map_result = result_expr(map_loop).unwrap();
                                        let filter_loop = filter_for(&iters, map_result, *init_builder.clone(), params, &(*cond), lb_sym.clone(), ub_sym.clone()).unwrap();
                                        let switch = switchfor_expr(vec![expr.clone(), filter_loop], 
                                                                    vec![Annotations::new(), Annotations::new()], 
                                                                    bld_sym, 
                                                                    lb_sym, 
                                                                    ub_sym).unwrap();

                                        // Create let expressions if we had any literal vectors
                                        let mut prev_expr = switch;
                                        for &(ref data, ref name) in data_names.iter() {
                                            prev_expr = let_expr(name.clone(), data.clone(), prev_expr)?;
                                        }
                                        return Ok((Some(prev_expr), false));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok((None, true))
    });
}

/// Generates the inner loop that performs the mapping operation
fn map_for(iters: &Vec<Iter>, 
           params: &Vec<Parameter>, 
           map_expr: Expr, 
           lb_sym: Symbol, 
           ub_sym: Symbol,
           par_syms: Vec<Symbol>) -> WeldResult<Expr> {

    if let Merge { ref value, .. } = map_expr.kind {
        let bld_ty = Appender(Box::new(value.ty.clone()));

        // Create builder expression
        let ub_expr = ident_expr(ub_sym, Scalar(ScalarKind::I64))?;
        let lb_expr = ident_expr(lb_sym, Scalar(ScalarKind::I64))?;
        let len_expr = binop_expr(BinOpKind::Subtract, ub_expr, lb_expr)?;
        let bld = newbuilder_expr(bld_ty, vec![len_expr])?;

        // Create new parameters with updated names and types
        let mut new_params = params.clone();
        new_params[0].ty = bld.ty.clone();
        let mut symbol_map = BTreeMap::new();
        for i in 0..3 {
            symbol_map.insert(new_params[i].name.clone(), par_syms[i].clone());
            new_params[i].name = par_syms[i].clone();
        }

        // Update ident expressions so that new parameters are referenced
        let mut map_expr = map_expr.clone();
        map_expr.ty = bld.ty.clone();
        map_expr.transform(&mut |ref mut e| {
            if let Ident(ref old_sym) = e.kind {
                if let Some(new_sym) = symbol_map.get(old_sym) {
                    return Some(ident_expr(new_sym.clone(), e.ty.clone()).unwrap());
                }
            }
            None
        });

        let mut lambda = lambda_expr(new_params, map_expr.clone())?;
        Ok(for_expr(iters.clone(), bld, lambda, false)?)
    } else {
        compile_err!("Internal error: Non merge expression given for map_expr in map_for")
    }    
}

/// Generates the outer loop that filters the mapped data
fn filter_for(iters: &Vec<Iter>, map_result: Expr, bld: Expr, params: &Vec<Parameter>, filter_cond: &Expr, lb_sym: Symbol, ub_sym: Symbol) -> WeldResult<Expr> {
    if let Vector(map_elm_ty) = map_result.ty.clone() {
        let mut types = iters.iter()
            .map(|i| (*i.clone().data).ty)
            .map(|t| {
                match t {
                    Vector(ty) => Some(ty),
                    _ => None
                }
            })
            .map(|s| *(s.unwrap()))
            .collect::<Vec<Type>>();
        types.push(*map_elm_ty.clone());
        let elm_ty = Struct(types);
        
        let mut new_iters = iters.clone();
        for i in new_iters.iter_mut() {
            i.start = Some(Box::new(ident_expr(lb_sym.clone(), Scalar(I64))?));
            i.end = Some(Box::new(ident_expr(ub_sym.clone(), Scalar(I64))?));
            i.stride = Some(Box::new(literal_expr(I64Literal(1))?));
        }
        let iter = Iter::new_simple(map_result);
        new_iters.push(iter);
        
        let mut new_params = params.clone();
        new_params[2].ty = elm_ty;

        let mut cond = filter_cond.clone();
        cond.transform_and_continue(&mut |ref mut e| {
            match e.kind {
                GetField { ref expr, .. } => {
                    if let Ident(ref sym) = expr.kind {
                        if *sym == params[2].name {
                            return (None, false);
                        }
                    }
                    (None, true)
                }
                Ident(ref sym) if *sym == params[2].name => {
                    (Some(param_field(&new_params, 2, 0)), false)
                }
                _ => (None, true)
            }
        });
        let on_true = merge_expr(param_ident(&new_params, 0), param_field(&new_params, 2, iters.len() as u32))?;
        let on_false = param_ident(&new_params, 0);
        let body = if_expr(cond, on_true, on_false)?;
        let lambda = lambda_expr(new_params, body)?;

        Ok(for_expr(new_iters, bld, lambda, false)?)
    } else {
        compile_err!("Internal error: Non vector expression given as map_result in filter_for")
    }
}

/// Transformation that removes vectors from for expressions when they aren't used.
pub fn remove_unused_for_data(expr: &mut Expr) {
    expr.transform_and_continue(&mut |ref mut expr| {
        if let For { ref iters, builder: ref init_builder, ref func } = expr.kind {
            if let Lambda { ref params, ref body } = func.kind {
                // First find which iters are used
                let elem_sym = &params[2].name;
                let mut used_indices = BTreeSet::new();
                let mut used_all = false;
                body.traverse_and_continue(&mut |ref mut e| {
                    if let GetField  { ref expr, index } = e.kind {
                        if let Ident(ref s) = expr.kind {
                            if s == elem_sym {
                                used_indices.insert(index);
                            }
                            return false;
                        }
                    }
                    if let Ident(ref s) = e.kind {
                        if s == elem_sym {
                            used_all = true;
                            return false;
                        }
                    }
                    true
                });

                // If not all iters were used, transform the for loop
                if !used_all && used_indices.len() < iters.len() && used_indices.len() > 0 {
                    if let Struct(_) = params[2].ty {
                        // Create new iterators
                        let mut new_iters = used_indices.iter().map(|idx| iters[*idx as usize].clone()).collect::<Vec<_>>();

                        // Create new element parameter
                        let mut data_types = vec![];
                        for iter in new_iters.iter() {
                            if let Vector(ref elem_ty) = iter.data.ty {
                                data_types.push(*elem_ty.clone());
                            }
                        }
                        let new_elem_param_ty = if data_types.len() > 1 {
                            Struct(data_types)
                        } else {
                            data_types[0].clone()
                        };
                        let mut new_params = params.clone();
                        new_params[2] = Parameter::new(new_params[2].name.clone(), new_elem_param_ty);

                        // Update get field expressions with new indices
                        let mut new_idx = 0;
                        let mut new_body = *body.clone();
                        let new_elem_is_struct = used_indices.len() > 1;
                        for old_idx in used_indices {
                            new_body.transform_and_continue(&mut |ref mut e| {
                                if let GetField { ref expr, index } = e.kind {
                                    if index == old_idx {
                                        if let Ident(ref s) = expr.kind {
                                            if s == elem_sym {
                                                if new_elem_is_struct {
                                                    return (Some(param_field(&new_params, 2, new_idx)), false);
                                                } else {
                                                    return (Some(param_ident(&new_params, 2)), false);
                                                }
                                            }
                                        }
                                    }
                                }
                                (None, true)
                            });
                            new_idx += 1;
                            new_body = new_body.clone();
                        }

                        // Create new for loop with updated iters, element parameter and function body
                        let mut new_func = lambda_expr(new_params, new_body).unwrap();
                        let vectorized = if let IterKind::SimdIter = new_iters[0].kind { true } else { false };
                        let mut new_for = for_expr(new_iters, *init_builder.clone(), new_func, vectorized).unwrap();
                        return (Some(new_for), true);
                    }
                }
            }
        }
        (None, true)
    });
}

// HELPER FUNCTIONS

fn param_ident(params: &Vec<Parameter>, param_idx: usize) -> Expr {
    let param = &params[param_idx];
    ident_expr(param.name.clone(), param.ty.clone()).unwrap()
}

fn param_field(params: &Vec<Parameter>, param_idx: usize, field_idx: u32) -> Expr {
    let param_ident = param_ident(params, param_idx);
    getfield_expr(param_ident, field_idx).unwrap()
}

fn valid_builder(expr: &Expr) -> Option<BuilderKind> {
    if let Builder ( ref bk, _ ) = expr.ty {
        match expr.kind {
            GetField { .. } | Ident { .. } | NewBuilder { .. } => return Some(bk.clone()),
            _ => return None
        };
    }
    None
}