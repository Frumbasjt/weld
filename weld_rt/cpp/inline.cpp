#include "runtime.h"
#include "bf.h"

__attribute__((always_inline))
extern "C" void weld_rt_print_i64(int64_t val) {
  printf("%lld\n", val);
}

__attribute__((always_inline))
extern "C" void *weld_rt_get_merger_at_index(void *m, int64_t size, int32_t i) {
  intptr_t ptr = reinterpret_cast<intptr_t>(m);
  ptr = (ptr + (CACHE_LINE - 1)) & CACHE_MASK;
  return reinterpret_cast<void *>(ptr + num_cache_blocks(size) * i * CACHE_LINE);
}

// a non-full task executes in correct serial order (and by the same thread)
// after its associated full task (and potentially other non-full tasks also
// assicated with the same full task), so it can simply write into its full
// task's piece (the cur_piece) without creating a new one
__attribute__((always_inline))
extern "C" vec_piece *weld_rt_cur_vb_piece(void *v, int32_t my_id) {
  vec_builder *vb = (vec_builder *)v;
  return (vec_piece *)weld_rt_get_merger_at_index(vb->thread_curs, sizeof(vec_piece), my_id);
}

__attribute__((always_inline))
extern "C" void weld_rt_bf_add(void *bf, const key_t item) {
  weld_bf *wbf = (weld_bf *)bf;
  wbf->filter->insert(&(*(wbf->filter_data))[0], item);
}

__attribute__((always_inline))
extern "C" bool weld_rt_bf_contains(void *bf, const key_t item) {
  weld_bf *wbf = (weld_bf *)bf;
  return wbf->filter->contains(&(*(wbf->filter_data))[0], item);
}