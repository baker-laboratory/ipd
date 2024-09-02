#include <curand_kernel.h>
#include <ipd_cuda_util.hpp>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace ipd {
namespace sampling {

void _tip_atom_placer_check_input(
    Tensor nblist, Tensor don, Tensor acc, Tensor tipxyz, Tensor tipdon, Tensor tipacc, py::dict params) {
  CHECK_INPUT(nblist);
  CHECK_INPUT(don);
  CHECK_INPUT(acc);
  CHECK_INPUT(tipxyz);
  CHECK_INPUT(tipdon);
  CHECK_INPUT(tipacc);
  if (nblist.dim() != 3) throw std::runtime_error("nblist must be 3D tensor");
  if (don.dim() != 3) throw std::runtime_error("don must have shape N,4,2");
  if (acc.dim() != 3) throw std::runtime_error("acc must have shape N,4,2");
  if (don.size(1) != 4) throw std::runtime_error("don must have shape N,4,2");
  if (don.size(2) != 2) throw std::runtime_error("don must have shape N,4,2");
  if (acc.size(1) != 4) throw std::runtime_error("acc must have shape N,4,2");
  if (acc.size(2) != 2) throw std::runtime_error("acc must have shape N,4,2");
  if (tipxyz.dim() != 2) throw std::runtime_error("tipxzy must be N,3 tensor");
  if (tipdon.dim() != 3) throw std::runtime_error("tipdon must have shape N,4,2");
  if (tipacc.dim() != 3) throw std::runtime_error("tipacc must have shape N,4,2");
  if (tipdon.size(1) != 4) throw std::runtime_error("tipdon must have shape N,4,2");
  if (tipdon.size(2) != 2) throw std::runtime_error("tipdon must have shape N,4,2");
  if (tipacc.size(1) != 4) throw std::runtime_error("tipacc must have shape N,4,2");
  if (tipacc.size(2) != 2) throw std::runtime_error("tipacc must have shape N,4,2");
}

py::tuple tip_atom_placer(
    Tensor vox, Tensor don, Tensor acc, Tensor tipxyz, Tensor tipdon, Tensor tipacc, py::dict params) {
  _tip_atom_placer_check_input(vox, don, acc, tipxyz, tipdon, tipacc, params);
  return py::make_tuple();
}

template <typename F> inline __device__ void quat_to_rot(Vec4<F> q, kMat44<F> R) {
  R(0, 0) = (q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]);
  R(0, 1) = 2 * q[1] * q[2] - 2 * q[0] * q[3];
  R(0, 2) = 2 * q[1] * q[3] + 2 * q[0] * q[2];
  R(1, 0) = 2 * q[1] * q[2] + 2 * q[0] * q[3];
  R(1, 1) = (q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]);
  R(1, 2) = 2 * q[2] * q[3] - 2 * q[0] * q[1];
  R(2, 0) = 2 * q[1] * q[3] - 2 * q[0] * q[2];
  R(2, 1) = 2 * q[2] * q[3] + 2 * q[0] * q[1];
  R(2, 2) = (q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]);
}

template <typename F, typename G> inline __device__ Vec4<F> rand_unit_quat(G *gen, F quat_height) {
  Vec4<F> vec;
  while (true) {
    if constexpr (std::is_same<G, curandStatePhilox4_32_10_t>()) {
      float4 rand = curand_normal4(gen);
      vec = Vec4<F>(rand.w, rand.x, rand.y, rand.z);
    } else if constexpr (std::is_same<G, curandStateMRG32k3a_t>() || std::is_same<G, curandStateXORWOW_t>()) {
      float2 randa = curand_normal2(gen);
      float2 randb = curand_normal2(gen);
      vec = Vec4<F>(randa.x, randa.y, randb.x, randb.y);
    } else {
      vec = Vec4<F>(curand_normal(gen), curand_normal(gen), curand_normal(gen), curand_normal(gen));
    }
    vec.normalize();
    vec[0] = fabs(vec[0]);
    if (vec[0] >= quat_height) return vec;
  }
}

template <typename F, typename G> inline __device__ Vec3<F> rand_vec_normal(G *gen) {
  if constexpr (std::is_same<G, curandStatePhilox4_32_10_t>()) {
    float4 rand = curand_normal4(gen);
    return Vec3<F>(rand.x, rand.y, rand.z);
  } else if constexpr (std::is_same<G, curandStateMRG32k3a_t>() || std::is_same<G, curandStateXORWOW_t>()) {
    float2 randa = curand_normal2(gen);
    float randb = curand_normal(gen);
    return Vec3<F>(randa.x, randa.y, randb);
  } else {
    return Vec3<F>(curand_normal(gen), curand_normal(gen), curand_normal(gen));
  }
}

template <typename F, typename G> inline __device__ Vec4<F> rand_unit_quat_proj(G *gen, F quat_height) {
  Vec3<F> vec = rand_vec_normal<F, G>(gen) * quat_height;
  Vec4<F> quat(1, vec[0], vec[1], vec[2]);
  return quat.normalized();
}

template <typename F, typename G> inline __device__ Vec4<F> rand_unit_quat_proj3(G *gen, F tanacosquatheight) {
  Vec4<F> vec;
  while (true) {
    if constexpr (std::is_same<G, curandStatePhilox4_32_10_t>()) {
      float4 rand = curand_uniform4(gen);
      vec = Vec4<F>(0, rand.w, rand.x, rand.y);
    } else {
      vec = Vec4<F>(0, curand_uniform(gen), curand_uniform(gen), curand_uniform(gen));
    }
    if (vec.squaredNorm() > 1) continue;
    F rad = tanacosquatheight;
    vec *= rad;
    vec[0] = 1;
    return vec.normalized();
  }
}

template <typename F, typename G> inline __device__ Vec3<F> rand_vec_uniform(G *gen) {
  Vec3<F> vec;
  while (true) {
    if constexpr (std::is_same<G, curandStatePhilox4_32_10_t>()) {
      float4 rand = curand_normal4(gen);
      vec = Vec3<F>(rand.x, rand.y, rand.z);
    } else if constexpr (std::is_same<G, curandStateMRG32k3a_t>() || std::is_same<G, curandStateXORWOW_t>()) {
      float2 randa = curand_normal2(gen);
      float randb = curand_normal(gen);
      vec = Vec3<F>(randa.x, randa.y, randb);
    } else {
      vec = Vec3<F>(curand_normal(gen), curand_normal(gen), curand_normal(gen));
    }
    if (vec.squaredNorm() <= 1) return vec;
  }
}

template <typename G, typename F>
__global__ void randx_kernal(kTen3<F> out,
                             unsigned long long seed,
                             kMat3<F> cart_mean,
                             kMat3<F> cart_sd,
                             bool cart_uniform,
                             F quat_height,
                             bool orinormal,
                             F tanacosquatheight) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= out.dimension(0)) return;
  G gen;
  curand_init(seed, 0, idx, &gen);
  kMat44<F> xform(&out(idx, 0, 0));
  Vec4<F> quat;
  if (orinormal) {
    quat = rand_unit_quat_proj<F>(&gen, quat_height);
  } else {
    if (quat_height > 0.85) { // empirically determined switchover
      quat = rand_unit_quat_proj3<F, G>(&gen, tanacosquatheight);
    } else {
      quat = rand_unit_quat<F>(&gen, quat_height);
    }
  }
  quat_to_rot(quat, xform);
  Vec3<F> vec;
  if (cart_uniform) {
    vec = rand_vec_uniform<F>(&gen);
  } else {
    vec = rand_vec_normal<F>(&gen);
  }
  // Vec3<F> = rand_unit_uniform<F>(&gen);
  xform(0, 3) = cart_mean[0] + cart_sd(0) * vec[0];
  xform(1, 3) = cart_mean[1] + cart_sd(1) * vec[1];
  xform(2, 3) = cart_mean[2] + cart_sd(2) * vec[2];
  xform(3, 3) = 1;
}

Tensor rand_xform(int64_t n,
                  Tensor cart_mean,
                  Tensor cart_sd,
                  bool cart_uniform,
                  float quat_height,
                  bool orinormal,
                  string dtype,
                  int nt = 256,
                  int64_t seed = std::numeric_limits<int64_t>::max(),
                  string gentype = "curandState") {
  CHECK_INPUT(cart_mean);
  CHECK_INPUT(cart_sd);
  Tensor out = at::zeros({n, 4, 4}, TensorOptions().dtype(str2dtype(dtype)).device(kCUDA));
  if (cart_sd.dim() != 1 || cart_sd.size(0) != 3) throw std::runtime_error("cart_sd must be 3D tensor");
  // out[out.size(0) - 1, 0, 0] = 1;
  float tanacosquatheight = tan(acos(quat_height));
  if (quat_height >= 1) throw std::runtime_error("quat_height must be less than 1");
  if (seed == std::numeric_limits<int64_t>::max()) seed = time(0);
  if ("curandState" == gentype)
    randx_kernal<curandState><<<n, nt>>>(kten3f(out), seed, kmat3f(cart_mean), kmat3f(cart_sd), cart_uniform,
                                         quat_height, orinormal, tanacosquatheight);
  // not implemented: curandStateScrambledSobol64_t curandStateSobol64_t curandStateScrambledSobol32_t
  // curandStateSobol32_t curandStateMtgp32_t curandStateMRG32k3a_t
  else if ("curandStatePhilox4_32_10_t" == gentype)
    randx_kernal<curandStatePhilox4_32_10_t><<<n, nt>>>(kten3f(out), seed, kmat3f(cart_mean), kmat3f(cart_sd),
                                                        cart_uniform, quat_height, orinormal,
                                                        tanacosquatheight);
  else if ("curandStateXORWOW_t" == gentype)
    randx_kernal<curandStateXORWOW_t><<<n, nt>>>(kten3f(out), seed, kmat3f(cart_mean), kmat3f(cart_sd),
                                                 cart_uniform, quat_height, orinormal, tanacosquatheight);
  else
    throw std::runtime_error("Unknown gentype " + gentype);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  return out;
}

template <typename T>
__global__ void
get_idx_of_items(T const *__restrict__ data, int ndata, int k, T val, int *__restrict__ out, int *outidx) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx > ndata) return;
  if (data[idx] < val) {
    int i = atomicAdd(outidx, 1);
    if (i >= k) return;
    out[i] = idx;
  }
}

template <typename T> Tensor sort_inplace_topk(Tensor data, int k) {
  CHECK_INPUT(data);
  assert(data.dim() == 1);
  Tensor orig = data.clone();
  thrust::device_ptr<T> beg = thrust::device_pointer_cast(data.data_ptr<T>());
  thrust::sort(beg, beg + data.size(0));
  Tensor out = at::empty({k}, TensorOptions().dtype(kInt).device(kCUDA));
  Tensor outidx = at::zeros({1}, TensorOptions().dtype(kInt).device(kCUDA));
  get_idx_of_items<T><<<data.size(0), 128>>>(orig.data_ptr<T>(), data.size(0), k, data[k].item<T>(),
                                             out.data_ptr<int>(), outidx.data_ptr<int>());
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  return out;
}

Tensor sort_inplace_topk_float(Tensor data, int k) { return sort_inplace_topk<float>(data, k); }

} // namespace sampling
} // namespace ipd
