#ifndef _WELD_RUNTIME_H_
#define _WELD_RUNTIME_H_

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <vector>
#include <dtl/bloomfilter/blocked_bloomfilter_logic.hpp>

// Weld dictionary interface.
#include "dict.h"
// Weld groupbuilder interface.
#include "groupbuilder.h"
// Weld bloomfilter interface.
#include "bf.h"

// flavor type
struct flavor_t;

// general profiling statistics for a task
typedef struct profile_stats_t {
  int64_t task_id;
  uint32_t calls;
  uint64_t tot_tuples;
  uint64_t tot_cycles;
  uint64_t prev_tuples;
  uint64_t prev_cycles;
  uint64_t start_cycle;
  uint64_t end_cycle;
} profile_stats_t;

// statistics for the vw-greedy algorithm
typedef struct vw_greedy_stats_t {
  int32_t flavor;
  int32_t explore_period;
  int32_t explore_length;
  int32_t exploit_period;
  int32_t phase_end;
  int32_t explore_start;
} vw_greedy_stats_t;

// work item
struct work_t {
  // task flavors
  flavor_t *flavors;
  // [lower, upper) gives the range of iteration indices for this task
  // it is [0, 0) if the task is not a loop body
  int64_t lower;
  int64_t upper;
  // updated in the user program -- the current iteration index for the outermost loop this task
  // covers, 0 if not a loop body task. Not updated in each iteration of innermost loops.
  int64_t cur_idx;
  // true if task was directly stolen from another queue or is a continuation.
  // If true, we need to set the nest_* fields so that we can create a new piece for
  // this task and its siblings in any associative builders.
  // Tasks that are not full tasks do not need separate nest_* fields because they
  // will be executed in serial order (and by the same thread)
  // after their associated full tasks and can share these full tasks' pieces
  // in any associative builders.
  int32_t full_task; // boolean
  // The list of loop indices of all containing loops for this task.
  // The whole program is assumed to be run in an outer loop with a single iteration.
  // This list is constructed by walking up the list of continuations of this task
  // and taking their cur_idx's. Indices for more outer loops will be earlier in the list.
  int64_t *nest_idxs;
  // For each entry of nest_idx, the task_id corresponding to the task from which it was obtained.
  // An algorithm for determining whether task A follows task B in the program's serial order
  // is as follows: compare A.nest_idxs to B.nest_idxs, starting from the low indices. If
  // two elements at the same index are different, the task with the lower element comes earlier
  // in the serial order. If the two elements are the same, compare the two elements at the same
  // position in nest_task_ids. If these two elements are different, the task with the lower
  // element comes earlier in the serial order, by the definition of task_id below. If the
  // two nest_task_ids elements are equal, proceed to the next nest_idxs index. (If the
  // nest_len's of the tasks are different, compare the arrays only up to the length of the
  // shorter one.) No two distinct tasks should be determined to be equal by this algorithm.
  int64_t *nest_task_ids;
  // length of nest_idxs and nest_task_ids
  int32_t nest_len;
  // If task B must execute after task A, and tasks A and B have identical nest_idxs, B.task_id
  // is guaranteed to be larger than A.task_id.
  int64_t task_id;
  // the continuation, NULL if no continuation
  work_t *cont;
  // if this task is a continuation, the number of remaining dependencies (0 otherwise)
  int32_t deps;
  // if this task has no continuation of its own, indicates whether program execution
  // was continued by the start of another loop at the end of this task
  int32_t continued; // boolean
  // largest task size (in # of iterations) that we should not split further (defaults to 0
  // for non-loop body tasks)
  int32_t grain_size;
  // number of function variants. 1 for regular tasks, higher for switchfor calls
  int32_t flavor_count;
  // some statistics for profiling
  profile_stats_t *profile;
  // some stastics specifically for the adaptive vw-greedy algorithm
  vw_greedy_stats_t *vw_greedy;
};

typedef struct work_t work_t;

// Weld parallel callback
typedef void (*par_func_t)(work_t*, int32_t);

typedef struct flavor_t {
  // parameters for the task function
  void *data;
  // pointer to the function
  par_func_t fp;
  // id of the function (needed for lazy compilation)
  int32_t func_id;
  // the globals that are updated after the flavor is run (if instrumented)
  void **instrumented_globals;
  // the number of globals that are updated. 0 means this is not an instrumented flavor.
  int32_t instrumented_globals_count;
  // the ids of defered assignments the flavor depends on, if any.
  int32_t *if_initialized;
  // the number of defered assignments the flavor depends on.
  int32_t if_initialized_count;
  // whether the function is currently being compiled (in case of lazy compilation)
  bool compiling;
  // number of times this flavor was called
  int32_t calls;
  // number of call when this flavor was last called
  int32_t last_called;
  // average cost of this flavor in cycles
  double avg_cost;
} flavor_t;

// VecBuilder structures
struct vec_piece {
  void *data;
  int64_t size;
  int64_t capacity;
  int64_t *nest_idxs;
  int64_t *nest_task_ids;
  int32_t nest_len;
};

struct vec_output {
  void *data;
  int64_t size;
};

struct vec_builder {
  std::vector<vec_piece> pieces;
  void *thread_curs;
  int64_t elem_size;
  int64_t starting_cap;
  bool fixed_size;
  void *fixed_vector;
  pthread_mutex_t lock;
};

extern "C" {
  void weld_runtime_init();

  // weld_rt functions can only be called from a runtime thread that is executing a Weld computation
  int32_t weld_rt_thread_id();
  void weld_rt_abort_thread();
  int32_t weld_rt_get_nworkers();
  int64_t weld_rt_get_run_id();

  void weld_rt_start_loop(work_t *w, void *body_data, void *cont_data, par_func_t body,
    par_func_t cont, int64_t lower, int64_t upper, int32_t grain_size);
  void weld_rt_start_switch(work_t *w, flavor_t *flavors, void *cont_data, par_func_t cont,
    int64_t lower, int64_t upper, int32_t grain_size, int32_t flavor_count);
  void weld_rt_set_result(void *res);

  void *weld_rt_new_vb(int64_t elem_size, int64_t starting_cap, int32_t fixed_size);
  void weld_rt_new_vb_piece(void *v, work_t *w, int32_t is_init_piece);
  vec_piece *weld_rt_cur_vb_piece(void *v, int32_t my_id);
  void weld_rt_set_vb_offset_if_fixed(void *v, int64_t offset);
  vec_output weld_rt_result_vb(void *v);

  void *weld_rt_new_merger(int64_t size, int32_t nworkers);
  void *weld_rt_get_merger_at_index(void *m, int64_t size, int32_t i);
  void weld_rt_free_merger(void *m);

  void weld_rt_defer_build(int32_t id, void (*condition)(bool*), par_func_t build, 
    void *build_params, void** depends_on);
  void *weld_rt_get_defered_result(int32_t id);
  void weld_rt_set_defered_result(int32_t id, void* result);

  par_func_t weld_rt_compile_func(void *module, int32_t func_id);

  // weld_run functions can be called both from a runtime thread and before/after a Weld computation is
  // executed
  int64_t weld_run_begin(par_func_t run, void* data, int64_t mem_limit, int32_t n_workers, void *module,
    int32_t explore_period, int32_t explore_length, int32_t exploit_period);
  void *weld_run_get_result(int64_t run_id);
  void weld_run_dispose(int64_t run_id);

  void *weld_run_malloc(int64_t run_id, size_t size);
  void *weld_run_realloc(int64_t run_id, void *data, size_t size);
  void weld_run_free(int64_t run_id, void *data);
  int64_t weld_run_memory_usage(int64_t run_id);

  int64_t weld_run_get_errno(int64_t run_id);
  void weld_run_set_errno(int64_t run_id, int64_t err);
}

// Helper defines for cache size
#define CACHE_BITS 6
#define CACHE_LINE (1 << CACHE_BITS)
#define CACHE_MASK (~(CACHE_LINE - 1))

inline int64_t num_cache_blocks(int64_t size) {
  // ceil of number of blocks
  return (size + (CACHE_LINE - 1)) >> CACHE_BITS;
}

#endif // _WELD_RUNTIME_H_
