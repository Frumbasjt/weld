use ast::*;
use ast::ExprKind::*;
use error::WeldResult;
use util::SymbolGenerator;

/// Normalize for expressions to that they are suitable to be included
/// in switchfor expressions.
pub fn adaptive_normalize_fors(expr: &mut Expr) {
    let mut sym_gen = SymbolGenerator::from_expression(&expr);
    expr.transform_and_continue(&mut |ref mut expr| {
        if let SwitchFor { .. } = expr.kind {
            return (None, false);
        }
        if let For { .. } = expr.kind {
            (extract_non_ident_iters(&mut expr.clone(), &mut sym_gen).unwrap(), true)
        } else {
            (None, true)
        }
    });
}

/// Check if the expression is an innermost For expression
pub fn is_inner_loop(expr: &mut Expr) -> bool {
    if let For { ref func, .. } = expr.kind {
        let mut is_inner = true;
        func.traverse_and_continue(&mut |ref mut e| {
            if let For { .. } = e.kind {
                is_inner = false;
            }
            is_inner
        });
        is_inner
    } else {
        false
    }
}

/// Check whether the expression is a switchfor, or a for loop with a nested switchfor.
pub fn contains_switchfor(expr: &Expr) -> bool {
    match expr.kind {
        SwitchFor { .. } => true,
        For { .. } => {
            let mut contains_switchfor = false;
            expr.traverse_and_continue(&mut |ref mut e| {
                if let SwitchFor { .. } = e.kind {
                    contains_switchfor = true;
                }
                !contains_switchfor
            });
            contains_switchfor
        },
        _ => false
    }
}

/// For a For expression, check if any of its data is not an identity expression. If so, replace them
/// with identity expressions, and prepend the for expression by corresponding let expressions.
pub fn extract_non_ident_iters(expr: &mut Expr, sym_gen: &mut SymbolGenerator) -> WeldResult<Option<Expr>> {
    if let For { ref iters, ref func, ref builder } = expr.kind {
        // Check if any of the iters is not an identity expression
        let mut contains_non_idents = false;
        for iter in iters.iter() {
            match iter.data.kind {
                Ident(_) => {},
                _ => {
                    contains_non_idents = true;
                    break;
                }
            }
        }
        if contains_non_idents {
            // Generate new iters with idents
            let mut data_names = Vec::new();
            let new_iters = iters.clone()
                .iter_mut()
                .map(|i| {
                    let mut i = i.clone();
                    // If the data is not an identity expression, replace it with one, and keep track of the name
                    // so that we can generate a let expression for it later
                    if let Ident(_) = (*i.data).kind { 
                        // Do nothing
                    } else {
                        let data = i.data.clone();
                        let data_name = sym_gen.new_symbol("a");
                        let data_ty = data.ty.clone();
                        i.data = Box::new(constructors::ident_expr(data_name.clone(), data_ty).unwrap());
                        data_names.push((*data, data_name));
                    }
                    i
                })
                .collect::<Vec<_>>();
            
            // Generate let expressions
            let mut prev_expr = constructors::for_expr(new_iters, *builder.clone(), *func.clone(), false)?;
            for &(ref data, ref name) in data_names.iter() {
                prev_expr = constructors::let_expr(name.clone(), data.clone(), prev_expr)?;
            }

            Ok(Some(prev_expr))
        } else {
            Ok(None)
        }
    } else {
        compile_err!("Internal error: extract_non_ident_iters expects For")
    }
}