#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" {
void _mlir_ciface_vector_add(MemRefDescriptor<float, 1> *a,
                             MemRefDescriptor<float, 1> *b,
                             MemRefDescriptor<float, 1> *out, int64_t n);
}

static int64_t parse_size(int argc, char **argv) {
  if (argc > 1) {
    return std::strtoll(argv[1], nullptr, 10);
  }
  if (const char *env = std::getenv("GPU_TUTORIAL_SIZE")) {
    return std::strtoll(env, nullptr, 10);
  }
  return 1024;
}

int main(int argc, char **argv) {
  const int64_t n = parse_size(argc, argv);
  if (n <= 0) {
    std::cerr << "size must be positive\n";
    return 2;
  }

  std::vector<float> a(n);
  std::vector<float> b(n);
  std::vector<float> out(n, -1.0f);

  for (int64_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i) * 0.5f;
    b[i] = static_cast<float>(i % 17) - 3.0f;
  }

  MemRefDescriptor<float, 1> a_ref{a.data(), a.data(), 0, {n}, {1}};
  MemRefDescriptor<float, 1> b_ref{b.data(), b.data(), 0, {n}, {1}};
  MemRefDescriptor<float, 1> out_ref{out.data(), out.data(), 0, {n}, {1}};

  _mlir_ciface_vector_add(&a_ref, &b_ref, &out_ref, n);

  double max_abs_error = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    const double expected = static_cast<double>(a[i] + b[i]);
    const double actual = static_cast<double>(out[i]);
    max_abs_error = std::max(max_abs_error, std::fabs(expected - actual));
  }

  std::cout << "MLIR GPU vector add n=" << n
            << " max_abs_error=" << std::setprecision(8) << max_abs_error
            << "\n";
  if (max_abs_error > 1.0e-6) {
    std::cerr << "vector_add output mismatch\n";
    return 1;
  }
  return 0;
}
