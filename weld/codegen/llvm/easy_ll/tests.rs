
use std::error::Error;

use codegen::llvm::easy_ll::compile_init_module;
use codegen::llvm::easy_ll::compile_and_add_to_module;

#[test]
fn basic_use() {
    let module = compile_init_module("
       define i64 @bar(i64 %arg) {
           %1 = add i64 %arg, 1
           ret i64 %1
       }
       define i64 @run(i64 %arg) {
           %1 = add i64 %arg, 1
           %2 = call i64 @bar(i64 %1)
           ret i64 %2
       }
    ", 2, false, None);
    assert!(module.is_ok());
    assert_eq!(module.unwrap().module.run(42), 44);
}

#[test]
fn add_module() {
    let module1 = compile_init_module("
        ; PRELUDE
        %foo_t = type i64

        define %foo_t @bar(%foo_t %arg) {
            %1 = add %foo_t %arg, 1
            ret %foo_t %1
        }

        ; BODY
        define i64 @run(i64 %arg) {
            %1 = bitcast i64 %arg to %foo_t
            %2 = add %foo_t %1, 1
            %3 = call i64 @bar(%foo_t %2)
            %4 = bitcast %foo_t %3 to i64
            ret i64 %4
        }
    ", 0, false, None);

    assert!(module1.is_ok());
    let module1 = module1.unwrap();
    assert_eq!(module1.module.run(42), 44);

    let func_and_code = compile_and_add_to_module::<i64, i64>(&module1.module, "
        ; PRELUDE
        %foo_t = type i64

        define %foo_t @bar(%foo_t %arg) {
            %1 = add %foo_t %arg, 1
            ret %foo_t %1
        }

        ; BODY
        define i64 @foo(i64 %arg) {
            %t.1 = bitcast i64 %arg to %foo_t
            %t.2 = call %foo_t @bar(%foo_t %t.1)
            %t.3 = bitcast %foo_t %t.2 to i64
            ret i64 %t.3
        }
    ", "foo", 0, false, None);

    assert!(func_and_code.is_ok());
    let (func, _) = func_and_code.unwrap();
    assert_eq!(func(42), 43);
}


#[test]
fn compile_error() {
    let module = compile_init_module("
       define ZZZZZZZZ @run(i64 %arg) {
           ret i64 0
       }
    ", 2, false, None);
    assert!(!module.is_ok());
    assert!(module.unwrap_err().description().contains("Compile"));
}

#[test]
fn no_run_function() {
    let module = compile_init_module("
       define i64 @ZZZZZZZ(i64 %arg) {
           ret i64 0
       }
    ", 2, false, None);
    assert!(!module.is_ok());
    assert!(module.unwrap_err().description().contains("run function"));
}

#[test]
fn wrong_function_type() {
    let module = compile_init_module("
       define i64 @run() {
           ret i64 0
       }
    ", 2, false, None);
    assert!(!module.is_ok());
    assert!(module.unwrap_err().description().contains("wrong type"));
}
