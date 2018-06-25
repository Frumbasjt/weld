use ast::*;
use ast::ExprKind::*;
use ast::Type::*;
use ast::constructors::*;
use util::SymbolGenerator;
use annotation::Annotations;
use annotation::IgnoreTransformKind;
use optimizer::transforms::adaptive_common::*;

pub fn adaptive_predication(expr: &mut Expr) {
    let mut sym_gen = SymbolGenerator::from_expression(&expr);
    expr.transform_and_continue_res(&mut |ref mut expr| {
        if expr.annotations.ignore_transforms().contains(&IgnoreTransformKind::AdaptivePredication) {
            return Ok((None, false));
        }
        if contains_switchfor(expr) {
            return Ok((None, false));
        }
        if let For { ref func, .. } = expr.kind {
            // Check if the for loop has an inner loop. If so, do not make it adaptive, but continue
            // to check if the inner loop may be adapted.
            let mut has_inner_loop = false;
            func.traverse(&mut |ref mut e| {
                if let For { .. } = e.kind {
                    has_inner_loop = true;
                }
            });
            if has_inner_loop {
                return Ok((None, true));
            }

            // Loop is inner loop, mark any candidates for predication
            let mut for_loop = expr.clone();
            let mut marked = 0;
            for_loop.transform_and_continue_res(&mut |ref mut e| {
                if e.annotations.adaptive_predication() {
                    return Ok((None, true));
                }
                if let If { ref on_true, ref on_false, .. } = e.kind {
                    if let Merge { ref builder, .. } = on_true.kind {
                        if let Ident(ref name) = on_false.kind {
                            if let Ident(ref name2) = builder.kind {
                                if name == name2 {
                                    if let Builder(ref bk, _) = builder.ty {
                                        match *bk {
                                            BuilderKind::Merger(_, _) |
                                            BuilderKind::DictMerger(_, _, _) | 
                                            BuilderKind::VecMerger(_, _) => {
                                                let mut res = e.clone();
                                                res.annotations.set_adaptive_predication(true);
                                                marked += 1;
                                                return Ok((Some(res), true));
                                            },
                                            _ => { }
                                        };
                                    }
                                }
                            }
                        }
                    } else {
                        let mut safe = true;
                        on_true.traverse(&mut |ref sub_expr| if sub_expr.kind.is_builder_expr() {
                            safe = false;
                        });
                        on_false.traverse(&mut |ref sub_expr| if sub_expr.kind.is_builder_expr() {
                            safe = false;
                        });
                        if safe {
                            if let Scalar(_) = on_true.ty {
                                if let Scalar(_) = on_false.ty {
                                    let mut res = e.clone();
                                    res.annotations.set_adaptive_predication(true);
                                    marked += 1;
                                    return Ok((Some(res), true));
                                }
                            }
                        }
                    }
                }
                Ok((None, true))
            });
            // If we found some candidates, transform the for into a switchfor
            if marked > 0 {
                let mut flavors = vec![];
                let mut flavors_annots = vec![];
                // Create a flavor for each combination of branching and predicated expressions
                for combination in get_combinations(marked) {
                    let mut flavor = for_loop.clone();
                    let mut i = 0;
                    flavor.transform(&mut |ref mut e| {
                        if e.annotations.adaptive_predication() {
                            let mut res = e.clone();
                            res.annotations.set_adaptive_predication(false);
                            res.annotations.set_predicate(combination[i]);
                            i += 1;
                            Some(res)
                        } else {
                            None
                        }
                    });
                    flavors.push(flavor);
                    flavors_annots.push(Annotations::new());
                }
                let switchfor = switchfor_expr(flavors, 
                                               flavors_annots,
                                               sym_gen.new_symbol("sw_bld"),
                                               sym_gen.new_symbol("lb_sym"),
                                               sym_gen.new_symbol("ub_sym")).unwrap();
                return Ok((Some(switchfor), false));
            }
        }
        Ok((None, true))
    });
}

// Get all possible combinations of if and select expressions for a given number of
// if expressions. A single combination is represented by a vector of bools, where
// the nth bool indicates whether the nth if expression should be predicated (true)
// or not (false).
fn get_combinations(num_ifs: u32) -> Vec<Vec<bool>> {
    let mut bit_vecs = vec![];
    // for i in 0..u32::pow(2, num_ifs as u32) {
    //     let mut bit_vec = vec![];
    //     for j in 0..num_ifs {
    //         let bit = i & (1 << j) != 0;
    //         bit_vec.push(bit);
    //     }
    //     bit_vecs.push(bit_vec);
    // }
    // For now limit the number of combinations because of code explosion
    bit_vecs.push(vec![false; num_ifs as usize]);
    bit_vecs.push(vec![true; num_ifs as usize]);
    return bit_vecs;
}