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
                                        // TODO: Perhaps only accept appenders, as the reorder may not be of any benefit otherwise (appending is expensive)

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
                                        // Check if start/stop/stride
                                        if iters.iter().any(|i| i.start.is_some()) {
                                            return Ok((None, true));
                                        }

                                        let mut sym_gen = SymbolGenerator::from_expression(&expr);
                                        let bld_sym = sym_gen.new_symbol("sw_bld");
                                        let lb_sym = sym_gen.new_symbol("lb");
                                        let ub_sym = sym_gen.new_symbol("ub");

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
                                        let map_loop = map_for(&iters, params, *on_true.clone(), lb_sym.clone(), ub_sym.clone()).unwrap();
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

/// Generates the loop that performs the mapping operation
fn map_for(iters: &Vec<Iter>, params: &Vec<Parameter>, map_expr: Expr, lb_sym: Symbol, ub_sym: Symbol) -> WeldResult<Expr> {
    if let Merge { ref value, .. } = map_expr.kind {
        let bld_ty = Appender(Box::new(value.ty.clone()));

        let ub_expr = ident_expr(ub_sym, Scalar(ScalarKind::I64))?;
        let lb_expr = ident_expr(lb_sym, Scalar(ScalarKind::I64))?;
        let len_expr = binop_expr(BinOpKind::Subtract, ub_expr, lb_expr)?;
        let bld = newbuilder_expr(bld_ty, vec![len_expr])?;

        let mut new_params = params.clone();
        new_params[0].ty = bld.ty.clone();
        let mut map_expr = map_expr.clone();
        map_expr.ty = bld.ty.clone();
        let lambda = lambda_expr(new_params, map_expr.clone())?;
        
        Ok(for_expr(iters.clone(), bld, lambda, false)?)
    } else {
        compile_err!("Internal error: Non merge expression given for map_expr in map_for")
    }    
}

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