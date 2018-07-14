#include "runtime.h"
#include "assert.h"
#include <algorithm>
#include <vector>
#include "dict.h"

extern "C" void *weld_rt_bf_new(int64_t num_items) {
  const size_t bf_size = num_items * 3;    // 3 bits per item
  bbf_t *bbf = new bbf_t(bf_size);
  std::vector<bf_word_t> *filter_data = new std::vector<bf_word_t>(bbf->get_length() / sizeof(bf_word_t));

  weld_bf *bf = (weld_bf *)weld_run_malloc(weld_rt_get_run_id(), sizeof(weld_bf));
  bf->filter = bbf;
  bf->filter_data = filter_data;

  return (void *)bf;
}

extern "C" void weld_rt_bf_free(void* bf) {
  weld_bf *wbf = (weld_bf *)bf;
  delete wbf->filter;
  delete wbf->filter_data;
  weld_run_free(weld_rt_get_run_id(), wbf);
}

extern "C" void weld_rt_bf_dict_insert(work_t *w, void *bf, void *d, void *cont_data, par_func_t cont) {
  uint32_t size = weld_rt_dict_size(d);
  if (size < 16384) {
    // Compute sequentially
    weld_bf *wbf = (weld_bf *)bf;
    wbf->filter->batch_insert(&((*wbf->filter_data)[0]), weld_rt_dict_hashes_to_array(d), size);

    // Run continuation
    // TODO: skip the runtime call altogether, this is so ugly
    flavor_t flav;
    flav.data = cont_data;
    work_t cont_task;
    cont_task.flavors = &flav;
    cont_task.flavor_count = 1;
    cont(&cont_task, 0);
  } else {
    // Body function data
    void **batch_insert_data = (void **)malloc(2 * sizeof(void*));
    batch_insert_data[0] = bf;
    batch_insert_data[1] = (void *)weld_rt_dict_hashes_to_array(d);  
    // Body function
    par_func_t batch_insert = [](work_t *t, int32_t f) {
      weld_bf *wbf = ((weld_bf **)t->flavors[0].data)[0];
      int32_t *hashes = ((int32_t **)t->flavors[0].data)[1];

      wbf->filter->batch_insert(&(*(wbf->filter_data))[0], hashes + t->lower, t->upper - t->lower);
    };
    // Start loop
    weld_rt_start_loop(w, batch_insert_data, cont_data, batch_insert, cont, 0, size, 16384);
  }
}