#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

const int64_t GPT2_VOCAB_SIZE = 50257;
const int64_t GPT2_EOS_TOKEN = 50256;

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" {
void _mlir_ciface_gpt2_decoder(MemRefDescriptor<float, 3> *output,
                               MemRefDescriptor<int64_t, 2> *input_ids,
                               MemRefDescriptor<int64_t, 2> *attention_mask);
}

std::vector<int64_t> generate_tokens(std::vector<int64_t> input_ids,
                                     std::vector<int64_t> attention_mask,
                                     int64_t max_new_tokens = 10) {
  if (attention_mask.empty()) {
    attention_mask.assign(input_ids.size(), 1);
  }

  for (int64_t step = 0; step < max_new_tokens; ++step) {
    int64_t seq_len = static_cast<int64_t>(input_ids.size());

    MemRefDescriptor<int64_t, 2> input_ids_memref = {
        input_ids.data(), input_ids.data(), 0, {1, seq_len}, {seq_len, 1}};
    MemRefDescriptor<int64_t, 2> attention_mask_memref = {
        attention_mask.data(),
        attention_mask.data(),
        0,
        {1, seq_len},
        {seq_len, 1}};
    MemRefDescriptor<float, 3> output_memref = {};

    _mlir_ciface_gpt2_decoder(
        &output_memref, &input_ids_memref, &attention_mask_memref);

    int64_t logits_offset = output_memref.offset;
    int64_t next_token = 0;
    float best_score = output_memref.aligned[logits_offset];
    for (int64_t token = 1; token < GPT2_VOCAB_SIZE; ++token) {
      float score = output_memref.aligned[logits_offset +
                                          token * output_memref.strides[2]];
      if (score > best_score) {
        best_score = score;
        next_token = token;
      }
    }

    std::free(output_memref.allocated);

    input_ids.push_back(next_token);
    attention_mask.push_back(1);

    if (next_token == GPT2_EOS_TOKEN) {
      break;
    }
  }

  return input_ids;
}

PYBIND11_MODULE(gpt_call, m) {
  m.def("generate_tokens", &generate_tokens,
        pybind11::arg("input_ids"), pybind11::arg("attention_mask"),
        pybind11::arg("max_new_tokens") = 10,
        "Greedy GPT-2 token generation");
}
