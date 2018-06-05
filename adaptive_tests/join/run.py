
import matplotlib.pyplot as plt
import yep
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
def generate_data(n_R, hit_S, hit_T, hit_U):
    R_a = np.arange(n_R, dtype='int64')
    R_z = np.arange(n_R, dtype='int64')

    n_S = int(hit_S * n_R)
    S_a = np.random.choice(R_a, n_S, replace=False) if n_R > 0 else np.empty(shape=0, dtype='int64')
    S_b = np.arange(n_S, dtype='int64')

    n_T = int(hit_T * n_S)
    T_b = np.random.choice(S_b, n_T, replace=False) if n_T > 0 else np.empty(shape=0, dtype='int64')
    T_c = np.arange(n_T, dtype='int64')

    n_U = int(hit_U * n_T)
    U_c = np.random.choice(T_c, n_U, replace=False) if n_U > 0 else np.empty(shape=0, dtype='int64')
    U_d = np.arange(n_U, dtype='int64')

    return [R_a, R_z, S_a, S_b, T_b, T_c, U_c, U_d]

# Perform the join in Python, check if hit ratios are accurate
def join_python(R_a, R_z, S_a, S_b, T_b, T_c, U_c, U_d):
    start = timer()

    S_ht = {}
    T_ht = {}
    U_ht = {}

    for (sa, sb) in zip(S_a, S_b):
        S_ht[sa] = sb

    for (tb, tc) in zip(T_b, T_c):
        T_ht[tb] = tc
        
    for (uc, ud) in zip(U_c, U_d):
        U_ht[uc] = ud

    aggregate = 0
    s_hit = 0.0
    s_try = 0.0
    t_hit = 0.0
    t_try = 0.0
    u_hit = 0.0
    u_try = 0.0
    hits = 0
    for (ra, rz) in zip(R_a, R_z):
        sb = S_ht.get(ra)
        s_try += 1.0
        if (sb != None):
            s_hit += 1.0
            tc = T_ht.get(sb)
            t_try += 1.0
            if (tc != None):
                t_hit += 1.0
                ud = U_ht.get(tc)
                u_try += 1.0
                if (ud != None):
                    u_hit += 1.0
                    hits += 1
                    aggregate += (ra + sb + tc + ud + rz)
    end = timer()

    print("S hit ratio: " + (str(s_hit / s_try) if s_try > 0 else str(0)))
    print("T hit ratio: " + (str(t_hit / t_try) if t_try > 0 else str(0)))
    print("U hit ratio: " + (str(u_hit / u_try) if u_try > 0 else str(0)))
    print("Hits: " + str(hits))

    return (aggregate, end - start)

# Create the args object for Weld
def args_factory(encoded):
    class Args(ctypes.Structure):
        _fields_ = [e for e in encoded]
    return Args 

# Join the tables using Weld, a number of times
def join_weld_times(values, adaptive, lazy, trials):
    last_result = None
    comp_times = np.array([])
    exec_times = np.array([])

    for i in range(trials):
        (result, comp_time, exec_time) = join_weld(values, adaptive, lazy)
        comp_times = np.append(comp_times, comp_time)
        exec_times = np.append(exec_times, exec_time)
        assert(last_result == None or last_result == result)
        last_result = result

    return (last_result, comp_times, exec_times)

# Join the tables using Weld
def join_weld(values, adaptive, lazy):
    weld_code = None
    with open('join.weld', 'r') as content_file:
        weld_code = content_file.read()

    enc = NumpyArrayEncoder()
    names = ['Ra', 'Rz', 'Sa', 'Sb', 'Tb', 'Tc', 'Uc', 'Ud']
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
    conf.set("weld.compile.dumpCode", "false")
    conf.set("weld.compile.traceExecution", "false")
    conf.set("weld.optimization.applyExperimentalTransforms", "true" if adaptive else "false")
    conf.set("weld.adaptive.lazyCompilation", "true" if lazy else "false")
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
    # yep.start(("lazy" if lazy else "adaptive" if adaptive else "normal") + ".prof")
    weld_ret = module.run(conf, arg, err)
    # yep.stop()
    exec_time = timer() - exec_start

    if err.code() != 0:
        raise ValueError(("Error while running function,\n{}\n\n"
                        "Error message: {}").format(
            weld_code, err.message()))

    ptrtype = POINTER(restype.ctype_class)
    data = ctypes.cast(weld_ret.data(), ptrtype)
    result = dec.decode(data, restype)
    
    weld_ret.free()
    arg.free()

    return (result, comp_time, exec_time)

# Program
data = generate_data(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
trials = int(sys.argv[5])

# print("Performing join in Python...")
# start = timer()
# (expected, python_time) = join_python(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7])
# python_time = timer() - start
# print("Result: " + str(expected))

print("Performing join in normal Weld...")
(normal_result, wnct, wnet) = join_weld_times(data, False, False, trials)
# print("Result: " + str(normal_result))
# assert(normal_result == expected)

print("Performing join in adaptive Weld...")
(adapt_result, wact, waet) = join_weld_times(data, True, False, trials)
# print("Result: " + str(adapt_result))
# assert(adapt_result == expected)

print("Performing join in lazy adaptive Weld...")
(lazy_result, wlct, wlet) = join_weld_times(data, True, True, trials)
# assert(lazy_result == expected)

# print("\nPython time: " + str(python_time))
print("\nWeld normal total time: " + str(wnct.mean() + wnet.mean()))
print("Weld normal compilation time: " + str(wnct.mean()))
print("Weld normal execution time: " + str(wnet.mean()))
print("\nWeld adaptive total time: " + str(wact.mean() + waet.mean()))
print("Weld adaptive compilation time: " + str(wact.mean()))
print("Weld adaptive execution time: " + str(waet.mean()))
print("\nWeld lazy total time: " + str(wlct.mean() + wlet.mean()))
print("Weld lazy compilation time: " + str(wlct.mean()))
print("Weld lazy execution time: " + str(wlet.mean()))
print("\nAdaptive total speedup: " + str((wnct.mean() + wnet.mean()) / (wact.mean() + waet.mean())))
print("Adaptive compilation speedup: " + str((wnct.mean()) / (wact.mean())))
print("Adaptive execution speedup: " + str((wnet.mean()) / (waet.mean())))
print("\nLazy total speedup: " + str((wnct.mean() + wnet.mean()) / (wlct.mean() + wlet.mean())))
print("Lazy compilation speedup: " + str((wnct.mean()) / (wlct.mean())))
print("Lazy execution speedup: " + str((wnet.mean()) / (wlet.mean())))

# Plot runtimes
compile_times = [wnct.mean(), wact.mean(), wlct.mean()];
exec_times = [wnet.mean(), waet.mean(), wlet.mean()];
compile_stds = [wnct.std(), wact.std(), wlct.std()];
exec_stds = [wnet.std(), waet.std(), wlet.std()];
width = 0.35
ind = np.arange(3)

p1 = plt.bar(ind, compile_times, width, yerr=compile_stds)
p2 = plt.bar(ind, exec_times, width, bottom=compile_times, yerr=exec_stds)

plt.ylabel('Time (seconds)')
plt.title('Runtime')
plt.xticks(ind, ('Normal', 'Adaptive', 'Lazy'))
# plt.legend((p1[0], p2[0]), ('Compile time', 'Execution time'))

plt.show(block=True)