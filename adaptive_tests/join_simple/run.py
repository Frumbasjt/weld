import numpy as np
import pandas as pd
import ctypes
from weld.weldobject import *
from weld.types import *
from weld.encoders import NumpyArrayEncoder, ScalarDecoder
import weld.bindings as cweld
from collections import namedtuple
from pprint import pprint
import sys
from timeit import default_timer as timer

# Create data
def generate_data(n_R, hit_S):
    R_a = np.arange(n_R, dtype='int64')
    R_z = np.arange(n_R, dtype='int64')

    n_S = int(hit_S * n_R)
    S_a = np.random.choice(R_a, n_S, replace=False) if n_R > 0 else np.empty(shape=0, dtype='int64')
    S_b = np.arange(n_S, dtype='int64')

    return [R_a, R_z, S_a, S_b]

# Perform the join in Python, check if hit ratios are accurate
def join_python(R_a, R_z, S_a, S_b):
    start = timer()

    S_ht = {}

    for (sa, sb) in zip(S_a, S_b):
        S_ht[sa] = sb

    aggregate = 0
    s_hit = 0.0
    s_try = 0.0
    hits = 0
    for (ra, rz) in zip(R_a, R_z):
        sb = S_ht.get(ra)
        s_try += 1.0
        if (sb != None):
            s_hit += 1.0
            aggregate += (ra + sb + rz)
    end = timer()

    print("S hit ratio: " + (str(s_hit / s_try) if s_try > 0 else str(0)))
    print("Hits: " + str(hits))

    return (aggregate, end - start)

# Create the args object for Weld
def args_factory(encoded):
    class Args(ctypes.Structure):
        _fields_ = [e for e in encoded]
    return Args 

# Join the tables using Weld
def join_weld(values, adaptive):
    weld_code = None
    with open('join_simple.weld', 'r') as content_file:
        weld_code = content_file.read()

    enc = NumpyArrayEncoder()
    names = ['Ra', 'Rz', 'Sa', 'Sb']
    argtypes = [enc.py_to_weld_type(x).ctype_class for x in values]
    encoded = [enc.encode(x) for x in values]

    Args = args_factory(zip(names, argtypes))
    weld_args = Args()
    for name, value in zip(names, encoded):
        setattr(weld_args, name, value)

    void_ptr = ctypes.cast(ctypes.byref(weld_args), ctypes.c_void_p)
    arg = cweld.WeldValue(void_ptr)

    # Compile the module
    err = cweld.WeldError()
    conf = cweld.WeldConf()
    conf.set("weld.compile.dumpCode", "true")
    conf.set("weld.compile.traceExecution", "false")
    conf.set("weld.optimization.applyExperimentalTransforms", "true" if adaptive else "false")
    conf.set("weld.llvm.optimization.level", "3")
    conf.set("weld.threads", "1")

    comp_start = timer()
    module = cweld.WeldModule(weld_code, conf, err)
    comp_time = timer() - comp_start

    if err.code() != 0:
        raise ValueError("Could not compile function {}: {}".format(
            weld_code, err.message()))

    # Run the module
    dec = ScalarDecoder()
    restype = WeldLong()
    err = cweld.WeldError()
    exec_start = timer()
    weld_ret = module.run(conf, arg, err)
    exec_time = timer() - exec_start
    if err.code() != 0:
        raise ValueError(("Error while running function,\n{}\n\n"
                        "Error message: {}").format(
            weld_code, err.message()))

    ptrtype = POINTER(restype.ctype_class)
    data = ctypes.cast(weld_ret.data(), ptrtype)
    result = dec.decode(data, restype)

    return (result, comp_time, exec_time)

# Program
data = generate_data(int(sys.argv[1]), float(sys.argv[2]))

print("Performing join in Python...")
start = timer()
(expected, python_time) = join_python(data[0], data[1], data[2], data[3])
python_time = timer() - start
# print("Result: " + str(expected))

print("Performing join in normal Weld...")
(normal_result, wnct, wnet) = join_weld(data, False)
# print("Result: " + str(normal_result))
assert(normal_result == expected)

print("Performing join in adaptive Weld...")
(adapt_result, wact, waet) = join_weld(data, True)
# print("Result: " + str(adapt_result))
assert(adapt_result == expected)

print("\nPython time: " + str(python_time))
print("\nWeld normal total time: " + str(wnct + wnet))
print("Weld normal compilation time: " + str(wnct))
print("Weld normal execution time: " + str(wnet))
print("\nWeld adaptive total time: " + str(wact + waet))
print("Weld adaptive compilation time: " + str(wact))
print("Weld adaptive execution time: " + str(waet))
print("\nAdaptive total speedup: " + str(float(wnct + wnet) / float(wact + waet)))
print("Adaptive compilation speedup: " + str(float(wnct) / float(wact)))
print("Adaptive execution speedup: " + str(float(wnet) / float(waet)))