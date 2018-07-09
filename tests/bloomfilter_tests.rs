//! Tests for runtime errors that Weld can throw.

extern crate libc;
extern crate weld;

mod common;
use common::*;

#[test]
fn bloomfilter_simple_contains() {
    let code = "||let bb1=bloombuilder[i32](100L);let bb2=merge(bb1,42);let bf=result(bb2);bfcontains(bf,42)";
    let ref conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const bool;
    let result = unsafe { *data };

    assert_eq!(result, true);
}

#[test]
fn empty_bloomfilter() {
    let code = "||let bb=bloombuilder[i32](100L);let bf=result(bb);bfcontains(bf,42)";
    let ref conf = default_conf();

    let ref input_data = 0;

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const bool;
    let result = unsafe { *data };

    assert_eq!(result, false);
}


#[test]
fn bloomfilter_batch_insert() {
    let code = "|k:vec[i32]|
                let d=result(for(k,dictmerger[i32,i32,+],|b,i,e|merge(b,{e,0})));
                let bf=result(bloombuilder[i32](len(d),d));
                result(for(k,merger[i32,+],|b,i,e|if(bfcontains(bf,e),merge(b,1),b)))";
    let ref conf = many_threads_conf();

    let size = 100000;
    let mut input_vec: Vec<i32> = vec![];
    for i in 0..size {
        input_vec.push(i as i32);
    }
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };
    assert_eq!(result, size);
}
