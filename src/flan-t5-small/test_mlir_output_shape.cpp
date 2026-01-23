#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <vector>

const int64_t MAX_DECODER_LEN = 32;
const int64_t VOCAB_SIZE = 32128;
const int64_t PAD_TOKEN = 0;

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" {
void _mlir_ciface_transformer_model(
    MemRefDescriptor<float, 2> *output,
    MemRefDescriptor<int64_t, 2> *input_ids,
    MemRefDescriptor<int64_t, 2> *attention_mask,
    MemRefDescriptor<int64_t, 2> *decoder_input_ids_padded,
    MemRefDescriptor<int64_t, 2> *decoder_attention_mask);
}

int main() {
  int64_t input_ids[1][10] = {{13959, 1566, 12, 2968, 10, 571, 33, 25, 58, 1}};
  int64_t encoder_attention_mask[1][10] = {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

  MemRefDescriptor<int64_t, 2> input_ids_MemRef = {
      (int64_t *)input_ids, (int64_t *)input_ids, 0, {1, 10}, {10, 1}};
  MemRefDescriptor<int64_t, 2> encoder_attention_mask_MemRef = {
      (int64_t *)encoder_attention_mask, (int64_t *)encoder_attention_mask, 0, {1, 10}, {10, 1}};

  int64_t decoder_input_ids_padded[1][MAX_DECODER_LEN];
  int64_t decoder_attention_mask[1][MAX_DECODER_LEN];
  
  for (int i = 0; i < MAX_DECODER_LEN; i++) {
    decoder_input_ids_padded[0][i] = PAD_TOKEN;
    decoder_attention_mask[0][i] = 0;
  }
  decoder_input_ids_padded[0][0] = PAD_TOKEN;
  decoder_attention_mask[0][0] = 1;

  // Try different output buffer sizes
  std::cout << "Testing output buffer allocation...\n\n";
  
  // Test 1: As 2D (1, vocab_size) - just one position
  {
    std::cout << "Test 1: Output as (1, " << VOCAB_SIZE << ")\n";
    std::vector<float> outputData(VOCAB_SIZE);
    std::fill(outputData.begin(), outputData.end(), -999.0f);

    MemRefDescriptor<int64_t, 2> decoder_ids_MemRef = {
        (int64_t *)decoder_input_ids_padded, (int64_t *)decoder_input_ids_padded, 0,
        {1, MAX_DECODER_LEN}, {MAX_DECODER_LEN, 1}};
    MemRefDescriptor<int64_t, 2> decoder_mask_MemRef = {
        (int64_t *)decoder_attention_mask, (int64_t *)decoder_attention_mask, 0,
        {1, MAX_DECODER_LEN}, {MAX_DECODER_LEN, 1}};

    MemRefDescriptor<float, 2> outputMemRef = {
        outputData.data(), outputData.data(), 0, 
        {1, VOCAB_SIZE}, 
        {VOCAB_SIZE, 1}};

    std::cout << "  Calling _mlir_ciface_transformer_model...\n";
    _mlir_ciface_transformer_model(
        &outputMemRef, &input_ids_MemRef, &encoder_attention_mask_MemRef,
        &decoder_ids_MemRef, &decoder_mask_MemRef);

    std::cout << "  Returned MemRef sizes: [" << outputMemRef.sizes[0] << ", " << outputMemRef.sizes[1] << "]\n";
    
    // Check if any valid values
    int valid_count = 0;
    for (int i = 0; i < VOCAB_SIZE; i++) {
      if (outputData[i] != -999.0f && !std::isnan(outputData[i])) valid_count++;
    }
    std::cout << "  Valid (non-sentinel) values: " << valid_count << "\n";
    if (valid_count > 0) {
      std::cout << "  ✓ SUCCESS - Output written to (1, vocab) buffer\n";
      // Print top 5
      std::vector<std::pair<float, int>> scores;
      for (int j = 0; j < VOCAB_SIZE; j++) {
        if (outputData[j] != -999.0f) scores.push_back({outputData[j], j});
      }
      std::partial_sort(scores.begin(), scores.begin() + 5, scores.end(),
          [](auto& a, auto& b) { return a.first > b.first; });
      std::cout << "  Top 5: ";
      for (int i = 0; i < 5; i++) 
        std::cout << scores[i].second << " ";
      std::cout << "\n";
    } else {
      std::cout << "  ✗ FAILED - No data written\n";
    }
  }
  
  std::cout << "\n";
  
  // Test 2: As 2D (1, seq_len*vocab_size) - all positions flattened
  {
    std::cout << "Test 2: Output as (1, " << (MAX_DECODER_LEN * VOCAB_SIZE) << ") [flattened 3D]\n";
    std::vector<float> outputData(MAX_DECODER_LEN * VOCAB_SIZE);
    std::fill(outputData.begin(), outputData.end(), -999.0f);

    MemRefDescriptor<int64_t, 2> decoder_ids_MemRef = {
        (int64_t *)decoder_input_ids_padded, (int64_t *)decoder_input_ids_padded, 0,
        {1, MAX_DECODER_LEN}, {MAX_DECODER_LEN, 1}};
    MemRefDescriptor<int64_t, 2> decoder_mask_MemRef = {
        (int64_t *)decoder_attention_mask, (int64_t *)decoder_attention_mask, 0,
        {1, MAX_DECODER_LEN}, {MAX_DECODER_LEN, 1}};

    MemRefDescriptor<float, 2> outputMemRef = {
        outputData.data(), outputData.data(), 0, 
        {1, MAX_DECODER_LEN * VOCAB_SIZE}, 
        {MAX_DECODER_LEN * VOCAB_SIZE, 1}};

    std::cout << "  Calling _mlir_ciface_transformer_model...\n";
    _mlir_ciface_transformer_model(
        &outputMemRef, &input_ids_MemRef, &encoder_attention_mask_MemRef,
        &decoder_ids_MemRef, &decoder_mask_MemRef);

    std::cout << "  Returned MemRef sizes: [" << outputMemRef.sizes[0] << ", " << outputMemRef.sizes[1] << "]\n";
    
    int valid_count = 0;
    for (size_t i = 0; i < outputData.size(); i++) {
      if (outputData[i] != -999.0f && !std::isnan(outputData[i])) valid_count++;
    }
    std::cout << "  Valid (non-sentinel) values: " << valid_count << "\n";
    if (valid_count > 0) {
      std::cout << "  ✓ SUCCESS - Output written to flattened buffer\n";
      
      // Check position 0 (first sequence position)
      std::cout << "  Position 0 top tokens: ";
      std::vector<std::pair<float, int>> scores;
      for (int j = 0; j < VOCAB_SIZE; j++) {
        if (outputData[j] != -999.0f) scores.push_back({outputData[j], j});
      }
      std::partial_sort(scores.begin(), scores.begin() + 5, scores.end(),
          [](auto& a, auto& b) { return a.first > b.first; });
      for (int i = 0; i < 5; i++) std::cout << scores[i].second << " ";
      std::cout << "\n";
    } else {
      std::cout << "  ✗ FAILED - No data written\n";
    }
  }

  return 0;
}