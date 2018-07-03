use ast::*;
use ast::ExprKind::*;
use ast::Type::*;

/// Matches for expression of the form: for(if(c, v1, v2), bld, func)
/// If v1 or v2 are both empty vector literals, transforms it to bld
/// If v1 is an empty vector literal, transforms it to if(c, bld, for(v2, bld, func))
/// If v2 is an empty vector literal, transforms it to if(c, for(v1, bld, func), b)
pub fn if_for(expr: &mut Expr) {
    expr.transform(&mut |ref mut e| {
        if let For { ref iters, ref builder, ref func } = e.kind {
            if iters.len() == 1 {
                if let If { ref cond, ref on_true, ref on_false } = iters[0].data.kind {
                    let mut empty_on_true = false;
                    let mut empty_on_false = false;
                    if let MakeVector { ref elems } = on_true.kind {
                        empty_on_true = elems.len() == 0;
                    }
                    if let MakeVector { ref elems } = on_false.kind {
                        empty_on_false = elems.len() == 0;
                    }
                    if empty_on_true && empty_on_false {
                        return Some(*builder.clone());
                    }
                    if empty_on_true || empty_on_false {
                        let new_iter_data = if empty_on_true { *on_false.clone() } else { *on_true.clone() };
                        let mut new_iter = iters[0].clone();
                        new_iter.data = Box::new(new_iter_data);
                        let new_for = constructors::for_expr(vec![new_iter], *builder.clone(), *func.clone(), false).unwrap();
                        let new_expr = if empty_on_true {
                            constructors::if_expr(*cond.clone(), *builder.clone(), new_for).unwrap()
                        } else {
                            constructors::if_expr(*cond.clone(), new_for, *builder.clone()).unwrap()
                        };
                        return Some(new_expr);
                    }
                }
            }
        }
        None
    });
}

/// Normalizes expressions of the form if(c, b, merge(b, e)) to if(c, merge(b, e), b).
/// This allows these expressions to be detected by the predication pass.
pub fn normalize_merge_ifs(expr: &mut Expr) {
    expr.transform(&mut |ref mut e| {
        if let If { ref cond, ref on_true, ref on_false } = e.kind {
            if is_builder_ident(on_true) && !is_builder_ident(on_false) {
                let new_cond = negate_cond(&(*cond));
                let mut new_if = constructors::if_expr(new_cond, *on_false.clone(), *on_true.clone()).unwrap();
                new_if.annotations = e.annotations.clone();
                return Some(new_if);
            }
        }
        None
    });
}

/// Normalizes conditions such that literals are always on the right side of an equals or
/// not equals operation.
/// Normalizes conditions with a binary operation on the left side of the form:
///     (binop == true) to (binop)
///     (binop != true) to (inverse(binop))
///     (binop == false) to (inverse(binop))
///     (binop != false) to (binop)
pub fn normalize_conditions(expr: &mut Expr) {
    expr.transform(&mut |ref mut e| {
        if let BinOp { ref kind, ref left, ref right } = e.kind {
            // Literal always on the right
            match *kind {
                BinOpKind::Equal |
                BinOpKind::NotEqual => {
                    if let Literal(_) = left.kind {
                        if let Literal(_) = right.kind {
                            // Do nothing if both are literals
                            return None;
                        } else {
                            return Some(constructors::binop_expr(kind.clone(), *right.clone(), *left.clone()).unwrap());
                        }
                    }
                }
                _ => { return None; }
            }
            // Remove redundant comparisons with boolean types
            if let Scalar(sk) = left.ty {
                if let ScalarKind::Bool = sk {
                    if *kind == BinOpKind::Equal {
                        if let Literal(ref kind) = right.kind {
                            if let LiteralKind::BoolLiteral(val) = kind {
                                if *val {
                                    return Some(*left.clone());
                                } else {
                                    return inverse_bool_binop(left);
                                }
                            }
                        }
                    }
                    if *kind == BinOpKind::NotEqual {
                        if let Literal(ref kind) = right.kind {
                            if let LiteralKind::BoolLiteral(val) = kind {
                                if !*val {
                                    return Some(*left.clone());
                                } else {
                                    return inverse_bool_binop(&(*left));
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    });
}

fn is_builder_ident(expr: &Expr) -> bool {
    if let Ident(_) = expr.kind {
        if let Builder(_, _) = expr.ty {
            return true;
        }
    }
    false
}

fn negate_cond(expr: &Expr) -> Expr {
    let false_lit = constructors::literal_expr(LiteralKind::BoolLiteral(false)).unwrap();
    constructors::binop_expr(BinOpKind::Equal, expr.clone(), false_lit).unwrap()
}

fn inverse_bool_binop(expr: &Expr) -> Option<Expr> {
    use ast::BinOpKind::*;
    if let BinOp { ref kind, ref left, ref right } = expr.kind {
        match kind {
            Equal => Some(constructors::binop_expr(NotEqual, *left.clone(), *right.clone()).unwrap()),
            NotEqual => Some(constructors::binop_expr(Equal, *left.clone(), *right.clone()).unwrap()),
            LessThan => Some(constructors::binop_expr(GreaterThanOrEqual, *left.clone(), *right.clone()).unwrap()),
            LessThanOrEqual => Some(constructors::binop_expr(GreaterThan, *left.clone(), *right.clone()).unwrap()),
            GreaterThan => Some(constructors::binop_expr(LessThanOrEqual, *left.clone(), *right.clone()).unwrap()),
            GreaterThanOrEqual => Some(constructors::binop_expr(LessThan, *left.clone(), *right.clone()).unwrap()),
            _ => None
        }
    } else {
        None
    }
}