#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

template <typename T> struct ScalarMemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
};

struct BufferEntry {
  bool is_scalar = false;
  std::vector<float> float_values;
  int64_t scalar_value = 0;
};

#define BN_DECL(prefix)                                                         \
  MemRefDescriptor<float, 1> *prefix##_mean,                                   \
      MemRefDescriptor<float, 1> *prefix##_var,                                \
      ScalarMemRefDescriptor<int64_t> *prefix##_tracked,

extern "C" {
void _mlir_ciface_resnet18(
    MemRefDescriptor<float, 2> *output, BN_DECL(b0) BN_DECL(b1) BN_DECL(b2)
        BN_DECL(b3) BN_DECL(b4) BN_DECL(b5) BN_DECL(b6) BN_DECL(b7)
            BN_DECL(b8) BN_DECL(b9) BN_DECL(b10) BN_DECL(b11) BN_DECL(b12)
                BN_DECL(b13) BN_DECL(b14) BN_DECL(b15) BN_DECL(b16)
                    BN_DECL(b17) BN_DECL(b18) BN_DECL(b19)
                        MemRefDescriptor<float, 4> *input);
}

#undef BN_DECL

std::vector<BufferEntry> load_buffers() {
  std::ifstream file("resnet18_buffers.csv");
  if (!file.is_open()) {
    throw std::runtime_error("could not open resnet18_buffers.csv");
  }

  std::vector<BufferEntry> buffers;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }

    std::stringstream ss(line);
    std::string kind;
    std::getline(ss, kind, ',');

    BufferEntry entry;
    if (kind == "i") {
      std::string value;
      std::getline(ss, value, ',');
      entry.is_scalar = true;
      entry.scalar_value = std::stoll(value);
      buffers.push_back(std::move(entry));
      continue;
    }

    entry.is_scalar = false;
    std::string value;
    while (std::getline(ss, value, ',')) {
      entry.float_values.push_back(std::stof(value));
    }
    buffers.push_back(std::move(entry));
  }

  return buffers;
}

MemRefDescriptor<float, 1> make_vector_memref(BufferEntry &entry) {
  if (entry.is_scalar) {
    throw std::runtime_error("expected float buffer, got scalar buffer");
  }
  int64_t length = static_cast<int64_t>(entry.float_values.size());
  return {entry.float_values.data(), entry.float_values.data(), 0, {length}, {1}};
}

ScalarMemRefDescriptor<int64_t> make_scalar_memref(BufferEntry &entry) {
  if (!entry.is_scalar) {
    throw std::runtime_error("expected scalar buffer, got float buffer");
  }
  return {&entry.scalar_value, &entry.scalar_value, 0};
}

#define BN_BLOCK(index)                                                         \
  &float_buffers[(index) * 2], &float_buffers[(index) * 2 + 1],                \
      &scalar_buffers[index]

int main() {
  auto raw_buffers = load_buffers();
  if (raw_buffers.size() != 60) {
    throw std::runtime_error("expected 60 ResNet buffer entries");
  }

  std::array<MemRefDescriptor<float, 1>, 40> float_buffers;
  std::array<ScalarMemRefDescriptor<int64_t>, 20> scalar_buffers;

  size_t float_index = 0;
  size_t scalar_index = 0;
  for (auto &entry : raw_buffers) {
    if (entry.is_scalar) {
      scalar_buffers.at(scalar_index++) = make_scalar_memref(entry);
    } else {
      float_buffers.at(float_index++) = make_vector_memref(entry);
    }
  }

  float input_data[1][3][224][224];
  float output_data[1][1000] = {};

  for (int k = 0; k < 3; k++) {
    for (int i = 0; i < 224; i++) {
      for (int j = 0; j < 224; j++) {
        input_data[0][k][i][j] = 1.0f;
      }
    }
  }

  MemRefDescriptor<float, 4> input_memref = {
      (float *)input_data,
      (float *)input_data,
      0,
      {1, 3, 224, 224},
      {3 * 224 * 224, 224 * 224, 224, 1}};

  MemRefDescriptor<float, 2> output_memref = {
      (float *)output_data,
      (float *)output_data,
      0,
      {1, 1000},
      {1000, 1}};

  _mlir_ciface_resnet18(&output_memref, BN_BLOCK(0), BN_BLOCK(1), BN_BLOCK(2),
                        BN_BLOCK(3), BN_BLOCK(4), BN_BLOCK(5), BN_BLOCK(6),
                        BN_BLOCK(7), BN_BLOCK(8), BN_BLOCK(9), BN_BLOCK(10),
                        BN_BLOCK(11), BN_BLOCK(12), BN_BLOCK(13),
                        BN_BLOCK(14), BN_BLOCK(15), BN_BLOCK(16),
                        BN_BLOCK(17), BN_BLOCK(18), BN_BLOCK(19),
                        &input_memref);

  float *output = output_memref.aligned;
  for (int64_t i = 0; i < output_memref.sizes[1]; ++i) {
    std::cout << std::fixed << std::setprecision(5) << output[i] << ' ';
  }
  std::cout << '\n';

  return 0;
}

#undef BN_BLOCK
