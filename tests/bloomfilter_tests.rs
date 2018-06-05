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