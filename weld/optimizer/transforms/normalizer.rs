use ast::*;
use ast::ExprKind::*;
use ast::Type::*;

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