#ifndef _WELD_BF_H_
#define _WELD_BF_H_

#include <stdint.h>
#include <dtl/bloomfilter/blocked_bloomfilter_logic.hpp>

using bf_key_t = int32_t;   // only 32-bit integer supported
using bf_word_t = int32_t;  // works best with AVX2/512
using bbf_t = dtl::blocked_bloomfilter_logic<bf_key_t, dtl::hasher, bf_word_t, 
                                             1 /*words per block*/, 1 /*sectors per words*/, 
                                             2 /*number of hash functions*/>;

struct weld_bf {
  bbf_t *filter;
  std::vector<bf_word_t> *filter_data;
};

extern "C" void *weld_rt_bf_new(int64_t num_items);

extern "C" void weld_rt_bf_free(void *bf);

extern "C" void weld_rt_bf_add(void *bf, const bf_key_t item);

extern "C" bool weld_rt_bf_contains(void *bf, const bf_key_t item);

#endif