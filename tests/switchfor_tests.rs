//! Tests for runtime errors that Weld can throw.

extern crate libc;
extern crate weld;

mod common;
use common::*;

#[test]
fn double_identical_switched_loops() {
    let code = "
        |x:vec[i32]|
        let bld=appender[i32];
        result(switchfor(
            |lb,ub|for(x,bld,|b,i,e|merge(b,e)),
            |lb,ub|for(x,bld,|b,i,e|merge(b,e))
        ))";
    let ref conf = default_conf();

    let mut input_vec: Vec<i32> = vec![];
    for i in 0..100000 {
        input_vec.push(i);
    }
    let ref input_data: WeldVec<i32> = WeldVec {
        data: input_vec.as_ptr(),
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const WeldVec<i32>;
    let result = unsafe { (*data).clone() };
    assert_eq!(result.len as usize, input_vec.len());
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) as i32 }, input_vec[i as usize])
    }
}

#[test]
fn switchfor_instrumented() {
    let code = "
        @(run_vars:@m=0.0)
        |x:vec[i32]|
        let bld=merger[i32,+];
        @(defered_until:@m>0.0)
        let one=1;
        result(switchfor(
            @(switch_instrumented:@m)
            |lb,ub|for(x,bld,|b,i,e|@(count_calls:@m)merge(b,1)),
            |lb,ub|for(x,bld,|b,i,e|merge(b,1)),
            @(switch_if_initialized:one)
            |lb,ub|for(x,bld,|b,i,e|merge(b,one))
        ))";
    let ref conf = default_conf();

    let mut input_vec: Vec<i32> = vec![];
    for i in 0..1000000 {
        input_vec.push(i);
    }
    let ref input_data: WeldVec<i32> = WeldVec {
        data: input_vec.as_ptr(),
        len: input_vec.len() as i64,
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };
    assert_eq!(result, 1000000);
}