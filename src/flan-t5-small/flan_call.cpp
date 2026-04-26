#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

const int64_t VOCAB_SIZE = 32128;
const int64_t PAD_TOKEN = 0;
const int64_t EOS_TOKEN = 1;
const int64_t MAX_DECODER_LEN = 32;

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" {
void _mlir_ciface_flan_decoder(
    MemRefDescriptor<float, 3> *output,
    MemRefDescriptor<int64_t, 2> *input_ids,
    MemRefDescriptor<int64_t, 2> *attention_mask,
    MemRefDescriptor<int64_t, 2> *decoder_input_ids);
}

std::vector<int64_t> generate_tokens(std::vector<int64_t> &input_ids,
                                     std::vector<int64_t> &attention_mask) {
  int64_t offset = 0;
  int64_t input_size = static_cast<int64_t>(input_ids.size());
  MemRefDescriptor<int64_t, 2> input_ids_memref = {
      input_ids.data(), input_ids.data(), offset, {1, input_size}, {input_size, 1}};
  MemRefDescriptor<int64_t, 2> encoder_attention_mask_memref = {
      attention_mask.data(),
      attention_mask.data(),
      offset,
      {1, input_size},
      {input_size, 1}};

  std::vector<int64_t> generated_tokens;
  generated_tokens.push_back(PAD_TOKEN);

  const int max_steps = 20;

  for (int step = 0; step < max_steps; step++) {
    int64_t current_len = static_cast<int64_t>(generated_tokens.size());
    if (current_len > MAX_DECODER_LEN) {
      break;
    }

    MemRefDescriptor<int64_t, 2> decoder_input_ids_memref = {
        generated_tokens.data(),
        generated_tokens.data(),
        0,
        {1, current_len},
        {current_len, 1}};

    MemRefDescriptor<float, 3> output_memref = {};

    _mlir_ciface_flan_decoder(
        &output_memref,
        &input_ids_memref,
        &encoder_attention_mask_memref,
        &decoder_input_ids_memref);

    int64_t returned_len = output_memref.sizes[1];
    int64_t last_pos = returned_len - 1;
    int64_t offset_to_last_pos =
        output_memref.offset + last_pos * output_memref.strides[1];

    std::vector<std::pair<float, int>> scores;
    scores.reserve(VOCAB_SIZE);
    for (int j = 0; j < VOCAB_SIZE; j++) {
      scores.push_back({output_memref.aligned[offset_to_last_pos +
                                              j * output_memref.strides[2]],
                        j});
    }

    std::free(output_memref.allocated);

    std::partial_sort(
        scores.begin(),
        scores.begin() + std::min<int>(5, scores.size()),
        scores.end(),
        [](const auto &a, const auto &b) { return a.first > b.first; });

    int64_t next_token = scores[0].second;
    generated_tokens.push_back(next_token);
    if (next_token == EOS_TOKEN) {
      break;
    }
  }

  return generated_tokens;
}

PYBIND11_MODULE(flan_call, m) {
  m.def("generate_tokens", &generate_tokens, "Greedy Flan-T5 token generation");
}
