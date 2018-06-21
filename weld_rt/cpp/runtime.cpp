#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <setjmp.h>
#include <map>
#include <queue>
#include <deque>
#include <algorithm>
#include <exception>
#include "assert.h"
#include "runtime.h"
#include <limits>

static inline uint64_t get_cycles() {
  uint64_t a, d;
  __asm volatile("rdtsc" : "=a" (a), "=d" (d));
  return a | (d<<32);
}

// These is needed to ensure each grain size is divisible by the SIMD vector size. A value of 64
// should be sufficiently high enough to protect against all the common vector lengths (4, 8,
// 16, 32, 64 - 64 is used for 8-bit values in AVX-512).
#define MAX_SIMD_SIZE   64

#ifdef __APPLE__
#include <sched.h>

typedef int pthread_spinlock_t;

static int pthread_spin_init(pthread_spinlock_t *lock, int pshared) {
    __asm__ __volatile__ ("" ::: "memory");
    *lock = 0;
    return 0;
}

static int pthread_spin_destroy(pthread_spinlock_t *lock) {
    return 0;
}

static int pthread_spin_lock(pthread_spinlock_t *lock) {
    while (1) {
        int i;
        for (i=0; i < 10000; i++) {
            if (__sync_bool_compare_and_swap(lock, 0, 1)) {
                return 0;
            }
        }
        sched_yield();
    }
}

static int pthread_spin_trylock(pthread_spinlock_t *lock) {
    if (__sync_bool_compare_and_swap(lock, 0, 1)) {
        return 0;
    }
    return 1;
}

static int pthread_spin_unlock(pthread_spinlock_t *lock) {
    __asm__ __volatile__ ("" ::: "memory");
    *lock = 0;
    return 0;
}
#endif

/*
The Weld parallel runtime. When the comments refer to a "computation",
this means a single complete execution of a Weld program.
*/
using namespace std;

typedef deque<work_t *> work_queue;
typedef pthread_spinlock_t work_queue_lock;

int64_t run_id = 0; // the current run ID
pthread_mutex_t global_lock;
// allows each thread to retrieve its run ID
pthread_key_t global_id;

pthread_once_t runtime_is_initialized = PTHREAD_ONCE_INIT;

typedef struct {
  int32_t id;
  void *result;
  void (*cond_func)(bool*);
  par_func_t build_func;
  void *build_params;
  void **depends_on;
} defered_assign;

struct run_data {
  pthread_mutex_t lock;
  int32_t n_workers;
  pthread_t *workers;
  work_queue *all_work_queues; // queue per worker
  work_queue_lock *all_work_queue_locks; // lock per queue
  jmp_buf *work_loop_roots; // jmp_buf per worker so that workers can exit easily on errors
  volatile bool done; // if a computation is currently running, have we finished it?
  void *result; // stores the final result of the computation to be passed back
  // to the caller
  int64_t mem_limit;
  map<intptr_t, int64_t> allocs;
  int64_t cur_mem;
  volatile int64_t err; // "errno" is a macro on some systems so we'll call this "err"
  map<int32_t, defered_assign*> defer_assign_data;
  void *module; // the Weld module
  int32_t explore_period;
  int32_t explore_length;
  int32_t exploit_period;
  FILE *profile_out;
};

map<int64_t, run_data*> *runs;

typedef struct {
  int64_t run_id;
  int32_t thread_id;
} thread_data;

void init() {
  pthread_mutex_init(&global_lock, NULL);
  pthread_key_create(&global_id, NULL);
  runs = new map<int64_t, run_data*>;
}

extern "C" void weld_runtime_init() {
  pthread_once(&runtime_is_initialized, init);
}

// *** weld_rt functions and helpers ***

extern "C" int32_t weld_rt_thread_id() {
  return reinterpret_cast<thread_data *>(pthread_getspecific(global_id))->thread_id;
}

extern "C" int64_t weld_rt_get_run_id() {
  return reinterpret_cast<thread_data *>(pthread_getspecific(global_id))->run_id;
}

static inline run_data *get_run_data_by_id(int64_t run_id) {
  pthread_mutex_lock(&global_lock);
  run_data *rd = runs->find(run_id)->second;
  pthread_mutex_unlock(&global_lock);
  return rd;
}

static inline run_data *get_run_data() {
  return get_run_data_by_id(weld_rt_get_run_id());
}

// set the result of the computation, called from generated LLVM
extern "C" void weld_rt_set_result(void *res) {
  get_run_data()->result = res;
}

extern "C" int32_t weld_rt_get_nworkers() {
  return get_run_data()->n_workers;
}

extern "C" void weld_rt_abort_thread() {
  longjmp(get_run_data()->work_loop_roots[weld_rt_thread_id()], 0);
}

static inline void set_nest(work_t *task) {
  assert(task->full_task);
  vector<int64_t> idxs;
  vector<int64_t> task_ids;
  idxs.push_back(task->cur_idx);
  task_ids.push_back(task->task_id);
  work_t *cur = task->cont;
  int32_t nest_len = 1;
  while (cur != NULL) {
    idxs.push_back(cur->cur_idx);
    // subtract 1 because the conts give us the continuations and we want the
    // task_id's of the loop bodies before the continuations
    task_ids.push_back(cur->task_id - 1);
    cur = cur->cont;
    nest_len++;
  }
  task->nest_idxs = (int64_t *)malloc(sizeof(int64_t) * nest_len);
  task->nest_task_ids = (int64_t *)malloc(sizeof(int64_t) * nest_len);
  task->nest_len = nest_len;
  // we want the outermost idxs to be the "high-order bits" when we do a comparison of task nests
  reverse_copy(idxs.begin(), idxs.end(), task->nest_idxs);
  reverse_copy(task_ids.begin(), task_ids.end(), task->nest_task_ids);
}

static inline void set_full_task(work_t *task) {
  // possible for task to already be full, e.g. if the head task from a start_loop call is queued
  // but stolen before it can be executed (the thief tries to set_full_task a second time)
  if (task->full_task) {
    return;
  }
  task->full_task = true;
  set_nest(task);
}

// attempt to steal from back of the queue of a random victim
// should be called when own work queue empty
static inline bool try_steal(int32_t my_id, run_data *rd) {
  int32_t victim = rand() % rd->n_workers;
  if (!pthread_spin_trylock(rd->all_work_queue_locks + victim)) {
    work_queue *its_work_queue = rd->all_work_queues + victim;
    if (its_work_queue->empty()) {
      pthread_spin_unlock(rd->all_work_queue_locks + victim);
      return false;
    } else {
      work_t *popped = its_work_queue->back();
      its_work_queue->pop_back();
      pthread_spin_unlock(rd->all_work_queue_locks + victim);
      set_full_task(popped);
      pthread_spin_lock(rd->all_work_queue_locks + my_id);
      (rd->all_work_queues + my_id)->push_front(popped);
      pthread_spin_unlock(rd->all_work_queue_locks + my_id);
      return true;
    }
  } else {
    return false;
  }
}

static inline work_t *last_cont(work_t *task) {
  work_t *last_cont = task;
  while (last_cont->cont != NULL) {
    last_cont = last_cont->cont;
  }
  return last_cont;
}

// set the continuation of w to cont and increment cont
// dependency count
static inline void set_cont(work_t *w, work_t *cont) {
  w->cont = cont;
  __sync_fetch_and_add(&cont->deps, 1);
}

static inline void insert_cont(work_t *w, work_t *cont) {
  if (w->cont != NULL) {
    last_cont(cont)->cont = w->cont;
  }
  w->cont = cont;
  __sync_fetch_and_add(&cont->deps, 1);
}

static inline work_t *clone_task(work_t *task) {
  work_t *clone = (work_t *)malloc(sizeof(work_t));
  memcpy(clone, task, sizeof(work_t));
  clone->full_task = false;
  return clone;
}

static inline work_t *new_task(int64_t id, flavor_t *flavors, int32_t n_flavors, int64_t lower, int64_t upper, int32_t grain_size, run_data *rd) {
  work_t *task = (work_t *)calloc(1, sizeof(work_t));
  task->flavors = flavors;
  task->flavor_count = n_flavors;
  task->lower = lower;
  task->upper = upper;
  task->cur_idx = lower;
  task->task_id = id,
  task->grain_size = grain_size;
  // TODO: only do this for n_flavors > 1
  task->vw_greedy = (vw_greedy_stats_t *)calloc(1, sizeof(vw_greedy_stats_t));
  if (n_flavors > 1) {
    task->vw_greedy->explore_period = rd->explore_period;
    task->vw_greedy->explore_length = rd->explore_length;
    task->vw_greedy->exploit_period = rd->exploit_period;
    task->vw_greedy->phase_end = task->vw_greedy->explore_length;
    task->vw_greedy->explore_start = 0;
  }
  task->profile = (profile_stats_t *)calloc(1, sizeof(profile_stats_t));
  task->profile->task_id = id;
  return task;
}

static inline work_t *new_task(int64_t id, par_func_t fp, void *data, int64_t lower, int64_t upper, int32_t grain_size, run_data *rd) {
  flavor_t *flavor = (flavor_t *)calloc(1, sizeof(flavor_t));
  flavor->fp = fp;
  flavor->data = data;
  return new_task(id, flavor, 1, lower, upper, grain_size, rd);
}

static inline work_t *new_task(int64_t id, par_func_t fp, void *data, run_data *rd) {
  return new_task(id, fp, data, 0, 0, 0, rd);
}

static inline bool defered_is_initialized(run_data *rd, int32_t defered_id) {
  return rd->defer_assign_data[defered_id]->result != NULL;
}

static inline bool defered_may_be_initialized(run_data *rd, int32_t defered_id) {
  if (!defered_is_initialized(rd, defered_id)) {
    bool result;
    rd->defer_assign_data[defered_id]->cond_func(&result);
    return result;
  }
  return true;
}

// Called after an instrumented task is finished. Checks if defered assignments are now
// ready to be built, and if so, creates a task for each one that is, and sets it
// as the continuation of the instrumented task. If necessary, a task is also created
// to compile lazy flavors, which is returned by the function.
static inline work_t *finish_instrumented(work_t *task, int32_t my_id, run_data *rd) {
  work_t *cont = task->cont;
  // Check for each defered assign whether it is ready to be built
  for (auto const& entry : rd->defer_assign_data) {
    const defered_assign *da = entry.second;
    if (da->result == NULL) {
      bool should_build = false;
      da->cond_func(&should_build);
      if (should_build) {
        work_t *build_task = new_task(-5, da->build_func, da->build_params, rd);
        insert_cont(task, build_task);
        cont->deps = 1;
        // build_task->cont = task->cont;
        // build_task->deps++;
        // task->cont = build_task;
      }
    }
  }

  // Check for each of the upcoming flavors whether we should compile them
  vector<flavor_t*> *to_compile = new vector<flavor_t*>();
  for (int32_t i = 0; i < cont->flavor_count; i++) {
    flavor_t *flavor = &cont->flavors[i];
    if (flavor->fp == NULL) {
      bool compile = true;
      for (int32_t i = 0; i < flavor->if_initialized_count; i++) {
        compile = defered_may_be_initialized(rd, flavor->if_initialized[i]);
        if (!compile) {
          break;
        }
      }

      if (compile) {
        // fprintf(stderr, "PUSHED FLAVOR TO COMPILE\n");
        to_compile->push_back(flavor);
      }
    }
  }

  if (!to_compile->empty()) {
    // Create compilation task
    work_t *compile_task = new_task(-10, [](work_t *t, int32_t f) { 
      vector<flavor_t*> *to_compile = (vector<flavor_t*> *)t->flavors[0].data;
      // TODO: change weld_rt_compile_func so it accepts a vector of functions to compile
      void *module = get_run_data()->module;
      for (auto const& flavor : *to_compile) {
        // fprintf(stderr, "started compiling %d\n", flavor->func_id);
        flavor->fp = weld_rt_compile_func(module, flavor->func_id); 
        // fprintf(stderr, "done compiling %d\n", flavor->func_id);
      }
      delete to_compile;
    }, (void *)to_compile, rd);

    return compile_task;
  }
  return NULL;
}

// Called once task function returns
// Decrease the dependency count of the continuation, run the continuation
// if necessary, or signal the end of the computation if we are done
static inline void finish_task(work_t *task, int32_t my_id, run_data *rd) {
  if (task->cont == NULL) {
    task->profile->end_cycle = get_cycles();
    if (rd->profile_out != NULL) {
      fprintf(rd->profile_out, "%lld,%lld,%lld\n", task->task_id, task->profile->start_cycle, task->profile->end_cycle);
    }
    if (!task->continued) {
      // if this task has no continuation and there was no for loop to end it,
      // the computation is over
      rd->done = true;
    }
    for (int i = 0; i < task->flavor_count; i++) {
      // free(task->flavors[i].data);
    }
    // free(task->profile);
    // free(task->flavors);
  } else {
    int32_t previous = __sync_fetch_and_sub(&task->cont->deps, 1);
    if (previous == 1) {
      task->profile->end_cycle = get_cycles();
      // If the task was instrumented, check if there are defered lets
      // to execute and flavors to compile.
      work_t *compile_task = NULL;
      if (task->flavors[0].instrumented_globals_count > 0) {
        compile_task = finish_instrumented(task, my_id, rd);
      }
      // Enqueue the continuation since we are the last dependency.
      // If there are any tasks left on our queue we know it's safe
      // to not make this full, since we must have executed all of
      // this task's predecessors in order without any of them having been stolen.
      pthread_spin_lock(rd->all_work_queue_locks + my_id);
      bool queue_empty = (rd->all_work_queues + my_id)->empty();
      pthread_spin_unlock(rd->all_work_queue_locks + my_id);
      if (queue_empty) {
        set_full_task(task->cont);
      }
      pthread_spin_lock(rd->all_work_queue_locks + my_id);
      (rd->all_work_queues + my_id)->push_front(task->cont);
      if (compile_task != NULL) {
        (rd->all_work_queues + my_id)->push_front(compile_task);
      }
      pthread_spin_unlock(rd->all_work_queue_locks + my_id);
      // we are the last sibling with this data, so we can free it
      for (int i = 0; i < task->flavor_count; i++) {
        // free(task->flavors[i].data);
      }
      // free(task->flavors);
      // fprintf(stderr, "finished task, moving on\n");
      if (rd->profile_out != NULL) {
        fprintf(rd->profile_out, "%lld,%lld,%lld\n", task->task_id, task->profile->start_cycle, task->profile->end_cycle);
      }
      if (task->flavor_count > 1) {
        for (int i = 0; i < task->flavor_count; i++) {
          break;
          fprintf(stderr, "Task %lld var %d: last_avg_cost=%f calls=%d\n", 
                          task->task_id, i, 
                          task->flavors[i].avg_cost,
                          task->flavors[i].calls);
        }
      }
    }
    // free(task->profile);
    if (task->full_task) {
      free(task->nest_idxs);
      free(task->nest_task_ids);
    }
  }
  free(task);
}

// called from generated code to schedule a for loop with the given body and continuation
// the data pointers store the closures for the body and continuation
// lower and upper give the iteration range for the loop
// w is the currently executing task
extern "C" void weld_rt_start_loop(work_t *w, void *body_data, void *cont_data, par_func_t body,
    par_func_t cont, int64_t lower, int64_t upper, int32_t grain_size) {

  flavor_t *flavors = (flavor_t *)malloc(sizeof(flavor_t));
  flavors->fp = body;
  flavors->data = body_data;

  weld_rt_start_switch(w, flavors, cont_data, cont, lower, upper, grain_size, 1);
}

extern "C" void weld_rt_start_switch(work_t *w, flavor_t *body_flavors, void *cont_data, 
    par_func_t cont, int64_t lower, int64_t upper, int32_t grain_size, int32_t flavor_count) {

  // fprintf(stderr, "weld_rt_start_switch\n");
  run_data *rd = get_run_data();
  int64_t next_task_id = w->task_id + 1;

  // If the body has an instrumented flavor, we'll make a separate task for it that 
  // must be run first. The instrumented flavor is always the first flavor.
  work_t *instrumented_task = NULL;
  if (flavor_count > 1 && body_flavors[0].instrumented_globals_count > 0 && upper - lower > grain_size) {
    // fprintf(stderr, "creating instrumented task\n");
    instrumented_task = new_task(next_task_id++, body_flavors, 1, lower, lower + grain_size, grain_size, rd);
    // The body task will skip the instrumented flavor.
    flavor_count--;
    flavor_t *tmp = (flavor_t *)malloc(sizeof(flavor_t) * flavor_count);
    memcpy(tmp, body_flavors + 1, sizeof(flavor_t) * flavor_count);
    body_flavors = tmp;
    lower += grain_size;
  }

  // Create task for the body, and set it as continuation for the instrumented version
  // if it exists.
  work_t *body_task = new_task(next_task_id++, body_flavors, flavor_count, lower, upper, grain_size, rd);
  if (instrumented_task != NULL) {
    set_cont(instrumented_task, body_task);
  }

  // Create task for final continuation of body, which inherits the current
  // task continuation.
  work_t *cont_task = new_task(next_task_id++, cont, cont_data, rd);
  cont_task->cur_idx = w->cur_idx;
  set_cont(body_task, cont_task);
  if (w != NULL) {
    if (w->cont != NULL) {
      // inherit the current task's continuation
      set_cont(cont_task, w->cont);
    } else {
      // this task has no continuation, but it has been effectively
      // continued by this loop so we don't want to end the computation
      // when this task completes
      w->continued = true;
    }
  }

  work_t *new_outer_task1 = NULL;
  work_t *new_outer_task2 = NULL;
  // If current task is a loop body with multiple iterations left, we want
  // to create tasks for the remaining iterations so that they execute after
  // the inner loop being created now (or are stolen by another thread, in which
  // case new nest data will be created so their execution order is unimportant).
  if (w != NULL && w->lower != w->upper && w->upper - w->cur_idx > 1) {
    new_outer_task1 = clone_task(w);
    new_outer_task1->lower = w->cur_idx + 1;
    new_outer_task1->cur_idx = w->cur_idx + 1;
    set_cont(new_outer_task1, w->cont) ;
    // always split into two tasks if possible
    if (new_outer_task1->upper - new_outer_task1->lower > 1) {
      new_outer_task2 = clone_task(new_outer_task1);
      int64_t mid = (new_outer_task1->lower + new_outer_task1->upper) / 2;
      new_outer_task1->upper = mid;
      new_outer_task2->lower = mid;
      new_outer_task2->cur_idx = mid;
      set_cont(new_outer_task2, w->cont);
    }
    // ensure that w immediately ends, and possible_new_outer_task contains whatever remained
    w->cur_idx = w->upper - 1;
  }

  int32_t my_id = weld_rt_thread_id();
  pthread_spin_lock(rd->all_work_queue_locks + my_id);
  if (new_outer_task2 != NULL) {
    // fprintf(stderr, "push new_outer_task2\n");
    (rd->all_work_queues + my_id)->push_front(new_outer_task2);
  }
  if (new_outer_task1 != NULL) {
    // fprintf(stderr, "push new_outer_task1\n");
    (rd->all_work_queues + my_id)->push_front(new_outer_task1);
  }
  if (instrumented_task != NULL) {
    // fprintf(stderr, "push instrumented_task %lld\n", instrumented_task->task_id);
    (rd->all_work_queues + my_id)->push_front(instrumented_task);
  } 
  else {
    // fprintf(stderr, "push body_task %lld\n", body_task->task_id);
    (rd->all_work_queues + my_id)->push_front(body_task);
  }
  pthread_spin_unlock(rd->all_work_queue_locks + my_id);
}

// repeatedly break off the second half of the task into a new task
// until the task's size in iterations drops below a certain threshold
static inline void split_task(work_t *task, int32_t my_id, run_data *rd) {
  while (task->upper - task->lower > task->grain_size) {
    work_t *last_half = clone_task(task);
    int64_t mid = (task->lower + task->upper) / 2;

    // The inner loop may be subject to vectorization, so modify the bounds to make the task size
    // divisible by the SIMD vector size.
    if (task->grain_size > 2 * MAX_SIMD_SIZE) {
      // Assumes that all vectorized inner loops have grain_size > 2 * MAX_SIMD_SIZE.
      mid = (mid / MAX_SIMD_SIZE) * MAX_SIMD_SIZE;
    }

    task->upper = mid;
    last_half->lower = mid;
    last_half->cur_idx = mid;
    // task must have non-NULL cont if it has non-zero number of iterations and therefore
    // is a loop body
    set_cont(last_half, task->cont);
    pthread_spin_lock(rd->all_work_queue_locks + my_id);
    (rd->all_work_queues + my_id)->push_front(last_half);
    pthread_spin_unlock(rd->all_work_queue_locks + my_id);
  }
}

static inline void cleanup_tasks_on_thread(work_t *cur_task, int32_t my_id, run_data *rd) {
  finish_task(cur_task, my_id, rd);
  while (true) {
    pthread_spin_lock(rd->all_work_queue_locks + my_id);
    if ((rd->all_work_queues + my_id)->empty()) {
      pthread_spin_unlock(rd->all_work_queue_locks + my_id);
      return;
    } else {
      work_t *popped = (rd->all_work_queues + my_id)->front();
      (rd->all_work_queues + my_id)->pop_front();
      pthread_spin_unlock(rd->all_work_queue_locks + my_id);
      finish_task(popped, my_id, rd);
    }
  }
}

static inline bool may_run(run_data *rd, flavor_t *flavor) {
  if (flavor->fp == NULL) {
    return false;
  }
  if (flavor->instrumented_globals_count > 0 && flavor->calls > 0) {
    return false;
  }
  if (flavor->if_initialized_count > 0) {
    for (size_t i = 0; i < flavor->if_initialized_count; i++) {
      if (!defered_is_initialized(rd, flavor->if_initialized[i])) {
        return false;
      }
    }
  }
  return true;
}

static inline bool is_init_phase(run_data *rd, work_t *task) {
  for (size_t i = 0; i < task->flavor_count; i++) {
    if (may_run(rd, &task->flavors[i]) && task->flavors[i].calls == 0) {
      return true;
    }
  }
  return false;
}

static inline int32_t get_lru_flavor(run_data *rd, work_t *task) {
  int32_t min_last_called = std::numeric_limits<int32_t>::max();
  int32_t flavor = -1;
  for (int32_t i = 0; i < task->flavor_count; i++) {
    if (may_run(rd, &task->flavors[i]) && task->flavors[i].last_called < min_last_called) {
      min_last_called = task->flavors[i].last_called;
      flavor = i;
    }
  }
  return flavor;
}

static inline int32_t get_best_flavor(run_data *rd, work_t *task) {
  double min_avg_cost = std::numeric_limits<double>::max();
  int32_t flavor = -1;
  for (int32_t i = 0; i < task->flavor_count; i++) {
    if (may_run(rd, &task->flavors[i]) && task->flavors[i].calls > 0 && task->flavors[i].avg_cost < min_avg_cost) {
      min_avg_cost = task->flavors[i].avg_cost;
      flavor = i;
    }
  }
  return flavor;
}

pthread_mutex_t vw_greedy_lock;
static inline void update_flavor(run_data *rd, work_t *task, int64_t cycles) {
  // fprintf(stderr, "calls: %d\tphase_end: %d\tflavor: %d\n", task->profile->calls, task->vw_greedy->phase_end, task->flavor);
  pthread_mutex_lock(&vw_greedy_lock);
  task->profile->tot_cycles += cycles;
  task->profile->tot_tuples += task->upper - task->lower;
  task->profile->calls++;
  task->flavors[task->flavor].calls++;
  task->flavors[task->flavor].last_called = task->profile->calls;

  if (task->flavor_count > 1 && task->profile->calls == task->vw_greedy->phase_end) {
    // TODO: there will be tasks that started before this happens,
    // but call update_flavor after this has happened. They should NOT contribute to
    // the average cost of this flavor. Perhaps work_t should get an additional field local_flavor
    // which indicates with which flavor it actually ran.
    profile_stats_t profile = *task->profile;
    vw_greedy_stats_t vw_greedy = *task->vw_greedy;
    task->flavors[task->flavor].avg_cost = (double) (task->profile->tot_cycles - task->profile->prev_cycles) /
                                            (task->profile->tot_tuples - task->profile->prev_tuples);

    if (task->profile->calls > task->vw_greedy->explore_start) {
      // fprintf(stderr, "EXPLORE\n");
      task->vw_greedy->explore_start += task->vw_greedy->explore_period;
      task->vw_greedy->explore_flavor = get_lru_flavor(rd, task);
      task->vw_greedy->explore_remaining = task->vw_greedy->explore_length;
      task->vw_greedy->phase_end += task->vw_greedy->explore_length;
    } else {
      // fprintf(stderr, "EXPLOIT\n");
      task->vw_greedy->best_flavor = get_best_flavor(rd, task);
      task->vw_greedy->phase_end += task->vw_greedy->exploit_period;
    }

    task->profile->prev_tuples = task->profile->tot_tuples;
    task->profile->prev_cycles = task->profile->tot_cycles;

    // fprintf(stderr, "task %llu: chose flavor %d\n", task->task_id, task->vw_greedy->flavor);
  }
  pthread_mutex_unlock(&vw_greedy_lock);
}

// keeps executing items from the work queue until it is empty
static inline void work_loop(int32_t my_id, run_data *rd) {
  while (true) {
    pthread_spin_lock(rd->all_work_queue_locks + my_id);
    if ((rd->all_work_queues + my_id)->empty()) {
      pthread_spin_unlock(rd->all_work_queue_locks + my_id);
      return;
    } else {
      work_t *popped = (rd->all_work_queues + my_id)->front();
      (rd->all_work_queues + my_id)->pop_front();
      pthread_spin_unlock(rd->all_work_queue_locks + my_id);
      split_task(popped, my_id, rd);
      // Exit the thread if there's an error.
      // We don't need to worry about freeing here; the runtime will
      // free all allocated memory as long as it is allocated with
      // `weld_run_malloc` or `weld_run_realloc`.
      if (rd->err != 0) {
        cleanup_tasks_on_thread(popped, my_id, rd);
        return;
      }
      if (!setjmp(rd->work_loop_roots[my_id])) {
        // Get variation to run
        // int32_t choice = get_function_variation(rd, popped);
        int32_t explore_remaining = popped->vw_greedy->explore_remaining;
        while (explore_remaining && !__sync_bool_compare_and_swap(&popped->vw_greedy->explore_remaining, explore_remaining, explore_remaining - 1)) {
          explore_remaining = popped->vw_greedy->explore_remaining;
        }
        int32_t choice = explore_remaining ? popped->vw_greedy->explore_flavor : popped->vw_greedy->best_flavor;
        popped->flavor = choice;

        // Run variation
        uint64_t start = get_cycles();
        __sync_bool_compare_and_swap(&popped->profile->start_cycle, 0, start);
        // fprintf(stdout, "thread %d: task %lld var %d | lo: %d hi %d\n", my_id, popped->task_id, choice, popped->lower, popped->upper);
        flavor_t flavor = popped->flavors[choice];
        flavor.fp(popped, choice);
        // popped->flavors[choice].fp(popped, choice);
        uint64_t end = get_cycles();
        
        // Update stats
        update_flavor(rd, popped, end - start);
        // update_function_variation_stats(rd, popped, choice, (end - start)*10);

        finish_task(popped, my_id, rd);
      } else {
        // error-case exit from task
        cleanup_tasks_on_thread(popped, my_id, rd);
        return;
      }
    }
  }
}

static void *thread_func(void *data) {
  pthread_setspecific(global_id, data);
  thread_data *td = reinterpret_cast<thread_data *>(data);
  pthread_mutex_lock(&global_lock);
  run_data *rd = runs->find(td->run_id)->second;
  pthread_mutex_unlock(&global_lock);

  int iters = 0;
  // this work_loop call is needed to complete any work items that are initially on the queue
  work_loop(td->thread_id, rd);
  while (!rd->done) {
    if (try_steal(td->thread_id, rd)) {
      iters = 0;
      work_loop(td->thread_id, rd);
    } else {
      // If this thread is stalling, periodically check for errors.
      iters++;
      if (iters > 1000000) {
        if (rd->err != 0) {
          break;
        }
        iters = 0;
      }
    }
  }
  free(data);
  return NULL;
}

extern "C" void weld_rt_defer_build(int32_t id, void (*condition)(bool*), 
  par_func_t build, void* build_params, void** depends_on) {

  defered_assign *da = (defered_assign *)malloc(sizeof(defered_assign));
  da->id = id;
  da->result = NULL;
  da->cond_func = condition;
  da->build_func = build;
  da->build_params = build_params;
  da->depends_on = depends_on;

  get_run_data()->defer_assign_data[id] = da;
}

extern "C" void *weld_rt_get_defered_result(int32_t id) {
  return get_run_data()->defer_assign_data[id]->result;
}

extern "C" void weld_rt_set_defered_result(int32_t id, void* result) {
  get_run_data()->defer_assign_data[id]->result = result;
}

// *** weld_run functions and helpers ***

// kick off threads running thread_func
// block until the computation is complete
extern "C" int64_t weld_run_begin(par_func_t run, void *data, int64_t mem_limit, int32_t n_workers, void *module,
    int32_t explore_period, int32_t explore_length, int32_t exploit_period) {

  run_data *rd = new run_data;
  pthread_mutex_init(&rd->lock, NULL);
  rd->n_workers = n_workers;
  rd->workers = new pthread_t[n_workers];
  rd->all_work_queue_locks = new work_queue_lock[n_workers];
  rd->all_work_queues = new work_queue[n_workers];
  rd->work_loop_roots = new jmp_buf[n_workers];
  rd->done = false;
  rd->result = NULL;
  rd->mem_limit = mem_limit;
  rd->cur_mem = 0;
  rd->err = 0;
  rd->module = module;
  rd->explore_period = explore_period;
  rd->explore_length = explore_length;
  rd->exploit_period = exploit_period;
  rd->profile_out = NULL;

  if (getenv("WELD_PROFILE") != NULL) {
    rd->profile_out = fopen("profile.csv", "w");
    fprintf(rd->profile_out, "task_id,start,finish\n");
  }

  work_t *run_task = new_task(0, run, data, rd);
  // this initial task can be thought of as a continuation
  set_full_task(run_task);
  rd->all_work_queues[0].push_front(run_task);

  int64_t my_run_id = __sync_fetch_and_add(&run_id, 1);
  pthread_mutex_lock(&global_lock);
  (*runs)[my_run_id] = rd;
  pthread_mutex_unlock(&global_lock);

  for (int32_t i = 0; i < rd->n_workers; i++) {
    pthread_spin_init(rd->all_work_queue_locks + i, 0);
  }

  for (int32_t i = 0; i < rd->n_workers; i++) {
    thread_data *td = (thread_data *)malloc(sizeof(thread_data));
    td->run_id = my_run_id;
    td->thread_id = i;
    if (rd->n_workers == 1) {
      thread_func(reinterpret_cast<void *>(td));
    } else {
      pthread_create(rd->workers + i, NULL, &thread_func, reinterpret_cast<void *>(td));
    }
  }

  if (rd->n_workers > 1) {
    for (int32_t i = 0; i < rd->n_workers; i++) {
      pthread_join(rd->workers[i], NULL);
    }
  }

  for (int32_t i = 0; i < n_workers; i++) {
    pthread_spin_destroy(rd->all_work_queue_locks + i);
  }
  delete [] rd->work_loop_roots;
  delete [] rd->all_work_queue_locks;
  delete [] rd->all_work_queues;
  delete [] rd->workers;

  if (rd->profile_out != NULL) {
    fclose(rd->profile_out);
  }

  return my_run_id;
}

extern "C" void *weld_run_malloc(int64_t run_id, size_t size) {
  run_data *rd = get_run_data_by_id(run_id);
  pthread_mutex_lock(&rd->lock);
  if (rd->cur_mem + size > rd->mem_limit) {
    pthread_mutex_unlock(&rd->lock);
    weld_run_set_errno(run_id, 7);
    weld_rt_abort_thread();
    return NULL;
  }
  rd->cur_mem += size;
  void *mem = malloc(size);
  rd->allocs[reinterpret_cast<intptr_t>(mem)] = size;
  pthread_mutex_unlock(&rd->lock);
  return mem;
}

extern "C" void *weld_run_realloc(int64_t run_id, void *data, size_t size) {
  run_data *rd = get_run_data_by_id(run_id);
  pthread_mutex_lock(&rd->lock);
  int64_t orig_size = rd->allocs.find(reinterpret_cast<intptr_t>(data))->second;
  if (rd->cur_mem - orig_size + size > rd->mem_limit) {
    pthread_mutex_unlock(&rd->lock);
    weld_run_set_errno(run_id, 7);
    weld_rt_abort_thread();
    return NULL;
  }
  rd->cur_mem -= orig_size;
  rd->allocs.erase(reinterpret_cast<intptr_t>(data));
  rd->cur_mem += size;
  void *mem = realloc(data, size);
  rd->allocs[reinterpret_cast<intptr_t>(mem)] = size;
  pthread_mutex_unlock(&rd->lock);
  return mem;
}

extern "C" void weld_run_free(int64_t run_id, void *data) {
  run_data *rd = get_run_data_by_id(run_id);
  pthread_mutex_lock(&rd->lock);
  rd->cur_mem -= rd->allocs.find(reinterpret_cast<intptr_t>(data))->second;
  rd->allocs.erase(reinterpret_cast<intptr_t>(data));
  free(data);
  pthread_mutex_unlock(&rd->lock);
}

extern "C" void *weld_run_get_result(int64_t run_id) {
  run_data *rd = get_run_data_by_id(run_id);
  return rd->err != 0 ? NULL : rd->result;
}

extern "C" int64_t weld_run_get_errno(int64_t run_id) {
  return get_run_data_by_id(run_id)->err;
}

extern "C" void weld_run_set_errno(int64_t run_id, int64_t err) {
  get_run_data_by_id(run_id)->err = err;
}

extern "C" int64_t weld_run_memory_usage(int64_t run_id) {
  return get_run_data_by_id(run_id)->cur_mem;
}

extern "C" void weld_run_dispose(int64_t run_id) {
  // printf("weld_run_dispose\n");
  run_data *rd = get_run_data_by_id(run_id);
  assert(rd->done);
  for (map<intptr_t, int64_t>::iterator it = rd->allocs.begin(); it != rd->allocs.end(); it++) {
    free(reinterpret_cast<void *>(it->first));
  }
  pthread_mutex_destroy(&rd->lock);
  delete rd;
  pthread_mutex_lock(&global_lock);
  runs->erase(run_id);
  pthread_mutex_unlock(&global_lock);
}