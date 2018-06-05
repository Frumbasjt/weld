; Template for a bloom filter, its builder type, and helper functions
;
; Parameters:
; - NAME: name to give generated type, without % or @ prefix
; - ELEM: LLVM type of the element
; - ELEM_PREFIX: prefix for helper functions of elem (e.g. @i32 or @MyStruct)

%{NAME} = type i8*
%{NAME}.bld = type i8*

; Create a new bloom filter builder
define %{NAME}.bld @{NAME}.bld.new(i64 %numItems) alwaysinline {{
    %builder = call i8* @weld_rt_bf_new(i64 %numItems)
    %builderTyped = bitcast i8* %builder to %{NAME}.bld
    ret %{NAME}.bld %builderTyped
}}

define %{NAME}.bld @{NAME}.bld.merge(%{NAME}.bld %bldPtr, {ELEM} %item) {{
    %rawHash = call i32 {ELEM_PREFIX}.hash({ELEM} %item)
    %finalizedHash = call i32 @hash_finalize(i32 %rawHash)
    %filterVoidPtr = bitcast %{NAME} %bldPtr to i8*
    call void @weld_rt_bf_add(i8* %filterVoidPtr, i32 %finalizedHash)
    ret %{NAME}.bld %bldPtr
}}

; Build the actual bloom filter (right now not actually doing anything)
define %{NAME} @{NAME}.bld.result(%{NAME}.bld %bldPtr) alwaysinline {{
    %result = bitcast %{NAME}.bld %bldPtr to %{NAME}
    ret %{NAME} %result
}}

; Check if a bloom filter contains a value
define i1 @{NAME}.contains(%{NAME} %filterPtr, {ELEM} %item) alwaysinline {{
    %filterVoidPtr = bitcast %{NAME} %filterPtr to i8*
    %rawHash = call i32 {ELEM_PREFIX}.hash({ELEM} %item)
    %finalizedHash = call i32 @hash_finalize(i32 %rawHash)
    %result = call i1 @weld_rt_bf_contains(i8* %filterVoidPtr, i32 %finalizedHash)
    ret i1 %result
}}
