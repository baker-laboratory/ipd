#pragma once

#include <ipd_util.hpp>

namespace ipd {

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                        \
  CHECK_CUDA(x);                                                                                              \
  CHECK_CONTIGUOUS(x)

inline std::ostream &operator<<(std::ostream &os, dim3 d) {
  os << "dim3(" << d.x << ',' << d.y << ',' << d.z << ")";
  return os;
}

#define gpuErrchk(ans)                                                                                        \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  } else {
    // fprintf(stderr, "cuda kernel finished successfully\n");
  }
}

#define PINGCUDA()                                                                                            \
  // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)                                               \
  printf("PING %s:%d %s\n", __FILE__, __LINE__, __func__)

inline void finish_kernel() {
  gpuErrchk(cudaGetLastError());
  cudaDeviceSynchronize();
}

template <typename I> __device__ std::tuple<I, I, I, bool> kernel_index3d(dim3 size) {
  I ix = blockIdx.x * blockDim.x + threadIdx.x;
  I iy = blockIdx.y * blockDim.y + threadIdx.y;
  I iz = blockIdx.z * blockDim.z + threadIdx.z;
  bool ok = ix < size.x && iy < size.y && iz < size.z;
  return std::tie(ix, iy, iz, ok);
}
template <typename I> __device__ std::tuple<I, I, bool> kernel_index2d(dim3 size) {
  I ix = blockIdx.x * blockDim.x + threadIdx.x;
  I iy = blockIdx.y * blockDim.y + threadIdx.y;
  bool ok = ix < size.x && iy < size.y;
  return std::tie(ix, iy, ok);
}
template <typename I> __device__ std::tuple<I, I, bool> kernel_index2d(I sx, I sy) {
  I ix = blockIdx.x * blockDim.x + threadIdx.x;
  I iy = blockIdx.y * blockDim.y + threadIdx.y;
  bool ok = ix < sx && iy < sy;
  return std::tie(ix, iy, ok);
}
template <typename I> __device__ std::tuple<I, bool> kernel_index1d(dim3 size) {
  I ix = blockIdx.x * blockDim.x + threadIdx.x;
  bool ok = ix < size.x;
  return std::tie(ix, ok);
}
template <typename I> __device__ std::tuple<I, bool> kernel_index1d(I size) {
  I ix = blockIdx.x * blockDim.x + threadIdx.x;
  bool ok = ix < size;
  return std::tie(ix, ok);
}

template <typename Sizes> std::tuple<dim3, dim3, dim3> blocks3d(Sizes sizes, Tensor nthread) {
  dim3 t(nthread[0].item<int>(), nthread[1].item<int>(), nthread[2].item<int>());
  dim3 s(sizes[0], sizes[1], sizes[2]);
  dim3 b(ceil((float)s.x / (float)t.x), ceil((float)s.y / (float)t.y), ceil((float)s.z / (float)t.z));
  return std::tie(t, b, s);
}
template <typename I> std::tuple<dim3, dim3, dim3> blocks2d(I s1, I s2, Tensor nthread) {
  dim3 t(nthread[0].item<int>(), nthread[1].item<int>(), 1);
  dim3 s(s1, s2, 1);
  dim3 b(ceil((float)s.x / (float)t.x), ceil((float)s.y / (float)t.y), 1);
  return std::tie(t, b, s);
}
template <typename Sizes> std::tuple<dim3, dim3, dim3> blocks3d(Sizes sizes, int nthread) {
  dim3 t(nthread, nthread, nthread);
  dim3 s(sizes[0], sizes[1], sizes[2]);
  dim3 b(ceil((float)s.x / (float)t.x), ceil((float)s.y / (float)t.y), ceil((float)s.z / (float)t.z));
  return std::tie(t, b, s);
}
template <typename Sizes> std::tuple<dim3, dim3, dim3> blocks2d(Sizes sizes, int nthread) {
  dim3 t(nthread, nthread, 1);
  dim3 s(sizes[0], sizes[1], 1);
  dim3 b(ceil((float)s.x / (float)t.x), ceil((float)s.y / (float)t.y), 1);
  return std::tie(t, b, s);
}
template <typename I> std::tuple<dim3, dim3, dim3> blocks1d(I size, int nthread) {
  dim3 t(nthread, 1, 1);
  dim3 s(size, 1, 1);
  dim3 b(ceil((float)s.x / (float)t.x), 1, 1);
  return std::tie(t, b, s);
}

template <typename T> inline __device__ char const *formatstr() { return "%f"; }
template <> inline __device__ char const *formatstr<float>() { return "%f "; }
template <> inline __device__ char const *formatstr<double>() { return "%f "; }
template <> inline __device__ char const *formatstr<int32_t>() { return "%d "; }
template <> inline __device__ char const *formatstr<int64_t>() { return "%lld "; }
template <> inline __device__ char const *formatstr<uint32_t>() { return "%d "; }
template <> inline __device__ char const *formatstr<uint64_t>() { return "%lld "; }

template <class T>
inline __device__ std::enable_if_t<std::is_arithmetic<T>::value>
kprint(char const *name, T val, char const *post = "\n") {
  printf("%s ", name);
  printf(formatstr<T>(), val);
  printf(post);
}

template <class T, int R, int C, int O, int W, int Z>
inline __device__ void kprint(char const *name, Eigen::Matrix<T, R, C, O, W, Z> mat, char const *post = "\n") {
  printf("%s Matrix<T, %i, %i>\n", name, R, C);
  for (int r = 0; r < mat.rows(); ++r) {
    for (int c = 0; c < mat.cols(); ++c) {
      printf(formatstr<T>(), mat(r, c));
    }
    printf(post);
  }
}
template <class T, int R, int C, int O, int W, int Z>
inline __device__ void kprint(char const *name, Eigen::Array<T, R, C, O, W, Z> mat, char const *post = "\n") {
  printf("%s Array<T, %i, %i>\n", name, R, C);
  for (int r = 0; r < mat.rows(); ++r) {
    for (int c = 0; c < mat.cols(); ++c) {
      printf(formatstr<T>(), mat(r, c));
    }
    printf(post);
  }
}
template <class T, int R, int C, int O, int W, int Z, int M, class S>
inline __device__ void
kprint(char const *name, Eigen::Map<Matrix<T, R, C, O, W, Z>, M, S> mat, char const *post = "\n") {
  printf("%s Map<Mat<T, %i, %i>>\n", name, R, C);
  for (int r = 0; r < mat.rows(); ++r) {
    for (int c = 0; c < mat.cols(); ++c) {
      printf(formatstr<T>(), mat(r, c));
    }
    printf(post);
  }
}
template <class T, int R, int O, int W, int Z, int M, class S>
inline __device__ void
kprint(char const *name, Eigen::Map<Matrix<T, R, 1, O, W, Z>, M, S> mat, char const *post = "\n") {
  printf("%s Map<Vec<T, %i>> ", name, R);
  for (int r = 0; r < mat.rows(); ++r) {
    printf(formatstr<T>(), mat[r]);
  }
  printf(post);
}
template <class T, int R, int O, int W, int Z>
inline __device__ void kprint(char const *name, Eigen::Matrix<T, R, 1, O, W, Z> mat, char const *post = "\n") {
  printf("%s Vec<T, %i> ", name, R);
  for (int r = 0; r < mat.rows(); ++r) {
    printf(formatstr<T>(), mat[r]);
  }
  printf(post);
}
// template <class T, int R, int O, int W, int Z>
// inline __device__ void kprint(char const *name, Eigen::Array<T, R, 1, O, W, Z> mat, char const *post = "\n")
// {
//   printf("%s Vec<T, %i> ", name, R);
//   for (int r = 0; r < mat.rows(); ++r) {
//     printf(formatstr<T>(), mat[r]);
//   }
//   printf(post);
// }

} // namespace ipd
