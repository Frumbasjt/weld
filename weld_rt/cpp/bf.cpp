#include "runtime.h"
#include "assert.h"
#include <algorithm>
#include <vector>

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