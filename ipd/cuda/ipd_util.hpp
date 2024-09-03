// Yeah, all this namespace pollution is ugly, but for these small extensino files,
// it's covenient

#pragma once

#include <Eigen/Dense>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <torch/extension.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace ipd {

namespace py = pybind11;
using namespace pybind11::literals;
using namespace torch;
using namespace torch::indexing;
using at::Half;
using Eigen::Dynamic;
using Eigen::Map;
using Eigen::Matrix;
using Eigen::RowMajor;
using Eigen::Vector3f;
using std::cerr;
using std::cout;
using std::endl;
using std::get;
using std::string;

#define PING() cerr << "PING " << __FILE__ << ":" << __LINE__ << " " << __func__ << endl;

// torch Tensor to Eigen Tensor Stuff
template <typename T, int ND> using kTen = Eigen::TensorMap<Eigen::Tensor<T, ND, RowMajor>>;
template <int ND> using kTenf = Eigen::TensorMap<Eigen::Tensor<float, ND, RowMajor>>;
template <int ND> using kTeni = Eigen::TensorMap<Eigen::Tensor<int, ND, RowMajor>>;
template <int ND> using kTenh = Eigen::TensorMap<Eigen::Tensor<Half, ND, RowMajor>>;
template <typename T> using kTen1 = Eigen::TensorMap<Eigen::Tensor<T, 1, RowMajor>>;
template <typename T> using kTen2 = Eigen::TensorMap<Eigen::Tensor<T, 2, RowMajor>>;
template <typename T> using kTen3 = Eigen::TensorMap<Eigen::Tensor<T, 3, RowMajor>>;
template <typename T> using kTen4 = Eigen::TensorMap<Eigen::Tensor<T, 4, RowMajor>>;
using kTen1f = kTenf<1>;
using kTen2f = kTenf<2>;
using kTen3f = kTenf<3>;
using kTen4f = kTenf<4>;
using kTen1h = kTenh<1>;
using kTen2h = kTenh<2>;
using kTen3h = kTenh<3>;
using kTen4h = kTenh<4>;
using kTen1i = kTeni<1>;
using kTen2i = kTeni<2>;
using kTen3i = kTeni<3>;
using kTen4i = kTeni<4>;
template <typename T> inline kTen<T, 1> kten1(Tensor t) {
  assert(t.dim() == 1);
  return kTen<T, 1>(t.data_ptr<T>(), t.size(0)); /*    */
}
template <typename T> inline kTen<T, 2> kten2(Tensor t) {
  assert(t.dim() == 2);
  return kTen<T, 2>(t.data_ptr<T>(), t.size(0), t.size(1)); /**/
}
template <typename T> inline kTen<T, 3> kten3(Tensor t) {
  assert(t.dim() == 3);
  return kTen<T, 3>(t.data_ptr<T>(), t.size(0), t.size(1), t.size(2));
}
template <typename T> inline kTen<T, 4> kten4(Tensor t) {
  assert(t.dim() == 4);
  return kTen<T, 4>(t.data_ptr<T>(), t.size(0), t.size(1), t.size(2), t.size(3));
}
// template <> inline kTen<Half, 1> kten1<Half>(Tensor t) {
// return kTen<Half, 1>((Half *)t.data_ptr<__half>(), t.size(0));
// }
// template <> inline kTen<Half, 2> kten2<Half>(Tensor t) {
// return kTen<Half, 2>((Half *)t.data_ptr<__half>(), t.size(0), t.size(1));
// }
// template <> inline kTen<Half, 3> kten3<Half>(Tensor t) {
// return kTen<Half, 3>((Half *)t.data_ptr<__half>(), t.size(0), t.size(1), t.size(2));
// }
// template <> inline kTen<Half, 4> kten4<Half>(Tensor t) {
// return kTen<Half, 4>((Half *)t.data_ptr<__half>(), t.size(0), t.size(1), t.size(2), t.size(3));
// }
inline kTen1f kten1f(Tensor t) { return kten1<float>(t); }
inline kTen2f kten2f(Tensor t) { return kten2<float>(t); }
inline kTen3f kten3f(Tensor t) { return kten3<float>(t); }
inline kTen4f kten4f(Tensor t) { return kten4<float>(t); }
inline kTen1h kten1h(Tensor t) { return kten1<Half>(t); }
inline kTen2h kten2h(Tensor t) { return kten2<Half>(t); }
inline kTen3h kten3h(Tensor t) { return kten3<Half>(t); }
inline kTen4h kten4h(Tensor t) { return kten4<Half>(t); }
inline kTen1i kten1i(Tensor t) { return kten1<int>(t); }
inline kTen2i kten2i(Tensor t) { return kten2<int>(t); }
inline kTen3i kten3i(Tensor t) { return kten3<int>(t); }
inline kTen4i kten4i(Tensor t) { return kten4<int>(t); }

// torch Tensor to Eigen Matrix Stuff (more efficient than tensor for <= 2D)
template <typename T, int N> using Vec = Eigen::Matrix<T, N, 1>;
template <typename T> using Vec1 = Vec<T, 1>;
template <typename T> using Vec2 = Vec<T, 2>;
template <typename T> using Vec3 = Vec<T, 3>;
template <typename T> using Vec4 = Vec<T, 4>;
using Vec1f = Vec1<float>;
using Vec2f = Vec2<float>;
using Vec3f = Vec3<float>;
using Vec4f = Vec4<float>;
using Vec1i = Vec1<int>;
using Vec2i = Vec2<int>;
using Vec3i = Vec3<int>;
using Vec4i = Vec4<int>;
template <typename T> inline Vec2<T> vec2(Tensor t) {
  assert(t.size(0) == 2);
  assert(t.dim() == 1);
  return Vec2<T>(t[0].item<T>(), t[1].item<T>());
}
template <typename T> inline Vec3<T> vec3(Tensor t) {
  assert(t.size(0) == 3);
  assert(t.dim() == 1);
  return Vec3<T>(t[0].item<T>(), t[1].item<T>(), t[2].item<T>());
}
template <typename T> inline Vec4<T> vec4(Tensor t) {
  assert(t.size(0) == 4);
  assert(t.dim() == 1);
  return Vec4<T>(t[0].item<T>(), t[1].item<T>(), t[2].item<T>(), t[3].item<T>());
}
inline Vec2f vec2f(Tensor t) { return vec2<float>(t); }
inline Vec3f vec3f(Tensor t) { return vec3<float>(t); }
inline Vec4f vec4f(Tensor t) { return vec4<float>(t); }
inline Vec2i vec2i(Tensor t) { return vec2<int>(t); }
inline Vec3i vec3i(Tensor t) { return vec3<int>(t); }
inline Vec4i vec4i(Tensor t) { return vec4<int>(t); }

template <typename T, int N> using Ary = Eigen::Array<T, N, 1>;
template <typename T> using Ary1 = Ary<T, 1>;
template <typename T> using Ary2 = Ary<T, 2>;
template <typename T> using Ary3 = Ary<T, 3>;
template <typename T> using Ary4 = Ary<T, 4>;
using Ary1f = Ary1<float>;
using Ary2f = Ary2<float>;
using Ary3f = Ary3<float>;
using Ary4f = Ary4<float>;
using Ary1i = Ary1<int>;
using Ary2i = Ary2<int>;
using Ary3i = Ary3<int>;
using Ary4i = Ary4<int>;
template <typename T> inline Ary2<T> ary2(Tensor t) {
  assert(t.size(0) == 2);
  assert(t.dim() == 1);
  return Ary2<T>(t[0].item<T>(), t[1].item<T>());
}
template <typename T> inline Ary3<T> ary3(Tensor t) {
  assert(t.size(0) == 3);
  assert(t.dim() == 1);
  return Ary3<T>(t[0].item<T>(), t[1].item<T>(), t[2].item<T>());
}
template <typename T> inline Ary4<T> ary4(Tensor t) {
  assert(t.size(0) == 4);
  assert(t.dim() == 1);
  return Ary4<T>(t[0].item<T>(), t[1].item<T>(), t[2].item<T>(), t[3].item<T>());
}
inline Vec2f ary2f(Tensor t) { return ary2<float>(t); }
inline Vec3f ary3f(Tensor t) { return ary3<float>(t); }
inline Vec4f ary4f(Tensor t) { return ary4<float>(t); }
inline Vec2i ary2i(Tensor t) { return ary2<int>(t); }
inline Vec3i ary3i(Tensor t) { return ary3<int>(t); }
inline Vec4i ary4i(Tensor t) { return ary4<int>(t); }

// eigen Matrix Maps
template <typename T, int R, int C> using kMat = Eigen::Map<Eigen::Matrix<T, R, C, RowMajor>>;
template <int R, int C> using kMatf = Eigen::Map<Eigen::Matrix<float, R, C, RowMajor>>;
template <typename T, int C> using kMatX = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, C, RowMajor>>;
template <typename T> using kMatX1 = Eigen::Map<Eigen::Matrix<T, Dynamic, 1>>;
template <typename T> using kMatX2 = kMatX<T, 2>;
template <typename T> using kMatX3 = kMatX<T, 3>;
template <typename T> using kMatX4 = kMatX<T, 4>;
using kMatX1f = kMatX1<float>;
using kMatX2f = kMatX2<float>;
using kMatX3f = kMatX3<float>;
using kMatX4f = kMatX4<float>;
using kMatX1i = kMatX1<int>;
using kMatX2i = kMatX2<int>;
using kMatX3i = kMatX3<int>;
using kMatX4i = kMatX4<int>;
template <typename T> using kMat1 = Eigen::Map<Eigen::Matrix<T, 1, 1>>;
template <typename T> using kMat2 = Eigen::Map<Eigen::Matrix<T, 2, 1>>;
template <typename T> using kMat3 = Eigen::Map<Eigen::Matrix<T, 3, 1>>;
template <typename T> using kMat4 = Eigen::Map<Eigen::Matrix<T, 4, 1>>;
using kMat1f = kMat1<float>;
using kMat2f = kMat2<float>;
using kMat3f = kMat3<float>;
using kMat4f = kMat4<float>;
using kMat1i = kMat1<int>;
using kMat2i = kMat2<int>;
using kMat3i = kMat3<int>;
using kMat4i = kMat4<int>;
template <typename T, int R, int C> inline kMat<T, R, C> kmat(Tensor t) {
  assert(t.size(0) == R);
  if constexpr (C == 1)
    assert(t.dim() == 1);
  else
    assert(t.size(1) == C);
  return kMat<T, R, C>(t.data_ptr<T>());
}
template <typename T, int C> inline kMatX<T, C> kmatx(Tensor t) {
  if constexpr (C == 1)
    assert(t.dim() == 1);
  else
    assert(t.size(1) == C);
  return kMatX<T, C>(t.data_ptr<T>(), t.size(0), C);
}
inline kMat1f kmat1f(Tensor t) { return kMat1f(t.data_ptr<float>()); }
inline kMat2f kmat2f(Tensor t) { return kMat2f(t.data_ptr<float>()); }
inline kMat3f kmat3f(Tensor t) { return kMat3f(t.data_ptr<float>()); }
inline kMat4f kmat4f(Tensor t) { return kMat4f(t.data_ptr<float>()); }
inline kMatX1f kmatx1f(Tensor t) { return kMatX1f(t.data_ptr<float>(), t.size(0)); }
inline kMatX2f kmatx2f(Tensor t) { return kmatx<float, 2>(t); }
inline kMatX3f kmatx3f(Tensor t) { return kmatx<float, 3>(t); }
inline kMatX4f kmatx4f(Tensor t) { return kmatx<float, 4>(t); }
inline kMatX1i kmatx1i(Tensor t) { return kMatX1i(t.data_ptr<int>(), t.size(0)); }
inline kMatX2i kmatx2i(Tensor t) { return kmatx<int, 2>(t); }
inline kMatX3i kmatx3i(Tensor t) { return kmatx<int, 3>(t); }
inline kMatX4i kmatx4i(Tensor t) { return kmatx<int, 4>(t); }
inline kMatX1<bool> kmatx1b(Tensor t) { return kMatX1<bool>(t.data_ptr<bool>(), t.size(0)); }

template <typename T> using Mat44 = Eigen::Matrix<T, 4, 4, RowMajor>;
template <typename T> using Mat33 = Eigen::Matrix<T, 3, 3, RowMajor>;
using Mat44f = Mat44<float>;
using Mat33f = Mat33<float>;
template <typename T> using kMat44 = Eigen::Map<Eigen::Matrix<T, 4, 4, RowMajor>>;
template <typename T> using kMat33 = Eigen::Map<Eigen::Matrix<T, 3, 3, RowMajor>>;
using kMat44f = kMat44<float>;
using kMat33f = kMat33<float>;
inline kMat44f kmat44f(Tensor t) { return kMat44f(t.data_ptr<float>()); }
inline kMat33f kmat33f(Tensor t) { return kMat33f(t.data_ptr<float>()); }
template <typename T> using kMat44C = Eigen::Map<Eigen::Matrix<T, 4, 4>>;
template <typename T> using kMat33C = Eigen::Map<Eigen::Matrix<T, 3, 3>>;
using kMat44Cf = kMat44C<float>;
using kMat33Cf = kMat33C<float>;
inline kMat44Cf kmat44Cf_WTF(Tensor t) { return kMat44Cf(t.data_ptr<float>()); }
inline kMat33Cf kmat33Cf_WTF(Tensor t) { return kMat33Cf(t.data_ptr<float>()); }

template <typename T> using kMatXT3 = Eigen::Map<Eigen::Matrix<Vec3<T>, Dynamic, Dynamic, RowMajor>>;
using kMatXf3 = kMatXT3<float>;
inline kMatXf3 kmatxf3(Tensor t) {
  assert(t.dim() == 3);
  assert(t.size(2) == 3);
  return kMatXf3((Vec3f *)t.data_ptr<float>(), t.size(0), t.size(1));
}

template <typename T> using kVecX = Eigen::Map<Eigen::Matrix<T, Dynamic, 1>>;
template <typename T> inline kVecX<T> kvecx(Tensor t) {
  assert(t.dim() == 1);
  return kVecX<T>(t.data_ptr<T>(), t.size(0));
}
inline kVecX<float> kvecxf(Tensor t) { return kvecx<float>(t); }
inline kVecX<int> kvecxi(Tensor t) { return kvecx<int>(t); }
inline kVecX<Vec3<float>> kvecx3f(Tensor t) {
  return kVecX<Vec3<float>>((Vec3<float> *)t.data_ptr<float>(), t.size(0));
}
using kVecXf = kVecX<float>;
using kVecXi = kVecX<int>;

template <typename T> inline T *make_device_ptr(T val) {
  T *devptr;
  cudaMalloc(&devptr, sizeof(T));
  cudaMemcpy(devptr, &val, sizeof(T), cudaMemcpyHostToDevice);
  return devptr;
}
template <typename T> inline T device_ptr_copy_and_free(T *devptr) {
  T result;
  cudaMemcpy(&result, devptr, sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(devptr);
  return result;
}

inline at::ScalarType str2dtype(std::string dtype) {
  if (dtype == "torch.float32") {
    return torch::kFloat32;
  } else if (dtype == "torch.float64") {
    return torch::kFloat64;
  } else if (dtype == "torch.int32") {
    return torch::kInt32;
  } else if (dtype == "torch.int64") {
    return torch::kInt64;
  } else if (dtype == "torch.bool") {
    return torch::kBool;
  } else {
    throw std::invalid_argument("Unsupported dtype: " + dtype);
  }
}

} // namespace ipd
