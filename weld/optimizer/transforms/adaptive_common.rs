use ast::*;
use ast::ExprKind::*;

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