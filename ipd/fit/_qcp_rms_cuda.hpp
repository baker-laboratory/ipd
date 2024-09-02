#include <ipd_cuda_util.hpp>
#include <thrust/device_vector.h>

namespace ipd {
namespace qcprms {

template <class F, bool calcrot = false>
__device__ inline F compute_rms(F const *__restrict__ A, F const E0, F const len, F *rot = nullptr);

namespace qcpscan_impl {
using Idx = Map<Matrix<int64_t, Dynamic, 1>>;

template <class I>
__device__ inline auto
qcp_scan_get_index(kMatX2i ranges, int64_t index, int chainbreak, I *buf, bool checkidx) {
  Idx idx(buf, ranges.rows());
  for (int j = 0; j < ranges.rows(); ++j) {
    int cumprod = 1;
    for (int i = j + 1; i < ranges.rows(); ++i)
      cumprod *= ranges(i, 1) - ranges(i, 0);
    idx(j) = ranges(j, 0) + (index / cumprod) % (ranges(j, 1) - ranges(j, 0));
    // printf("idx %i %i\n", j, idx(j));
  }
  bool invalid = false;
  if (checkidx)
    for (int ir = 0; ir < ranges.rows(); ++ir)
      for (int jr = ir + 1; jr < ranges.rows(); ++jr)
        if (idx(ir) == idx(jr)) invalid = true;
  if (chainbreak) {
    int nchain0 = 0; //(idx.array() < chainbreak).sum();  // why this no work???
    for (int i = 0; i < ranges.rows(); ++i)
      if (idx[i] < chainbreak) ++nchain0;
    if (nchain0 != (ranges.rows() / 2) && nchain0 != (ranges.rows() - ranges.rows() / 2)) invalid = true;
    // printf("nchain0 %d rows %d half %d half2 %d *%d* %d %d %d %d | %d %d\n", int(nchain0),
    // int(ranges.rows()), int(ranges.rows() / 2), int(ranges.rows() - ranges.rows() / 2), int(invalid),
    // int(idx[0]), int(idx[1]), int(idx[2]), int(idx[3]), ranges.rows() / 2 == nchain0, (ranges.rows() -
    // ranges.rows() / 2) == nchain0);
  }
  return std::tie(idx, invalid);
}

// why can't these be templates on F? compiler won't resolve
inline __device__ Vec3f getvec3(kMatXf3 dat, int i, int j, int k) { return dat(j, k); }
inline __device__ int getsize(kMatXf3 dat, int i) {
  if (i == 0) return dat.rows();
  if (i == 1) return dat.cols();
  if (i == 2) return 3;
  return -1;
}
inline __device__ Vec3f getvec3(kTen4f dat, int i, int j, int k) {
  return Vec3f(dat(i, j, k, 0), dat(i, j, k, 1), dat(i, j, k, 2));
}
inline __device__ int getsize(kTen4f dat, int i) { return dat.dimension(i + 1); }

template <class F, class BB> __device__ inline Vec3<F> get_center(BB bb, Idx idx, int Ncyc, int lasu) {
  lasu = lasu == 0 ? getsize(bb, 0) / Ncyc : lasu;
  Vec3<F> bbcen(0, 0, 0);
  for (int ir = 0; ir < idx.rows(); ++ir)
    for (int ia = 0; ia < getsize(bb, 1); ++ia)
      for (int ic = 0; ic < Ncyc; ++ic) {
        // kprint("bb ", getvec3(bb, ir, (int)idx(ir) + ic * lasu, ia), "\n");
        bbcen += getvec3(bb, ir, (int)idx(ir) + ic * lasu, ia);
      }
  bbcen /= idx.rows() * getsize(bb, 1) * Ncyc;
  // kprint("bbcen ", bbcen, "\n");
  return bbcen;
}

template <class F, class BB>
__device__ inline auto
qcp_scan_get_iprod_E0(BB bb, kMatXT3<F> tgt, Vec3<F> bbcen, Idx idx, int Ncyc, int lasu, F E0tgt) {
  lasu = lasu == 0 ? getsize(bb, 0) / Ncyc : lasu;
  Matrix<F, 3, 3, RowMajor> iprod = Matrix<F, 3, 3, RowMajor>::Zero();
  F E0 = E0tgt;
  for (int ir = 0; ir < idx.rows(); ++ir)
    for (int ia = 0; ia < getsize(bb, 1); ++ia)
      for (int ic = 0; ic < Ncyc; ++ic) {
        Vec3<F> p = getvec3(bb, ir, idx(ir) + ic * lasu, ia) - bbcen;
        iprod += p * tgt(ir + ic * idx.rows(), ia).transpose();
        E0 += p.array().square().sum();
      }
  F npts = (F)(idx.rows() * getsize(bb, 1) * Ncyc);
  return std::tie(iprod, E0, npts);
}

template <class F>
__device__ inline void qcp_scan_store_result(F rms, int64_t index, unsigned long long int *result) {
  unsigned long long int rms_as_int = fmax(0, rms) * 10000.0;
  unsigned long long int rmsidx = (rms_as_int << 44) | (unsigned long long int)index;
  // printf("rmsidx2 %lld %f %i %lld\n", rmsidx, rms, rms_as_int, index);
  if (rmsidx < *result) {
    // auto prev = *result;
    atomicMin(result, rmsidx);
    // printf("BEST atomicMin %lld -> %lld\n", prev, *result);
  }
}

template <class F, class BB>
__global__ void qcp_scan_rmsd_cuda_kernel(BB bb,
                                          kMatXT3<F> tgt,
                                          kMatX2i ranges,
                                          int Ncyc,
                                          int chainbreak,
                                          int lasu,
                                          F E0tgt,
                                          int64_t ntotal,
                                          bool checkidx,
                                          F *__restrict__ rmsout,
                                          unsigned long long int *result) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= ntotal) return;
  Idx::Scalar buf[16];
  auto [idx, invalid] = qcp_scan_get_index(ranges, index, chainbreak, buf, checkidx);
  if (invalid) return;
  auto bbcen = get_center<F, BB>(bb, idx, Ncyc, lasu);
  auto [iprod, E0, npts] = qcp_scan_get_iprod_E0<F, BB>(bb, tgt, bbcen, idx, Ncyc, lasu, E0tgt);
  // kprint("bbcen ", bbcen);
  // kprint("iprod ", iprod);
  // kprint("E0 ", E0);
  // kprint("npts ", npts);
  F rms = compute_rms<F>(iprod.data(), E0 / 2.0, npts);
  if (rmsout) rmsout[index] = rms;
  qcp_scan_store_result(rms, index, result);
}

auto qcp_scan_cuda_check_inputs(Tensor bb, Tensor tgt, Tensor ranges, int cyclic, int chainbreak, int lasu) {
  CHECK_INPUT(bb)
  CHECK_INPUT(tgt)
  CHECK_INPUT(ranges)
  if (bb.dim() == 2) bb = bb.unsqueeze(1);
  if (tgt.dim() == 2) tgt = tgt.unsqueeze(1);
  if (bb.dim() != 3 and bb.dim() != 4) throw std::runtime_error("qcp_scan_cuda: bb.ndim must be 3 or 4");
  for (int i = 0; i < ranges.size(0); ++i) {
    if (bb.dim() == 3 && bb.size(0) < cyclic * (ranges[i][1].item<int>() - ranges[i][0].item<int>()))
      throw std::runtime_error("qcp_scan_cuda: bb.ndim==3 and bb.size(0) < cyclic*ranges[i][1-0]");
    if (bb.dim() == 4 && bb.size(1) < cyclic * (ranges[i][1].item<int>() - ranges[i][0].item<int>()))
      throw std::runtime_error("qcp_scan_cuda: bb.ndim==4 and bb.size(1) > cyclic*ranges[i][1-0]");
  }
  if (bb.dim() == 3 && bb.size(0) < cyclic * lasu && lasu > 0)
    throw std::runtime_error("qcp_scan_cuda: bb.ndim==3 bb.size(0) < cyclic*lasu");
  if (bb.dim() == 4 && bb.size(1) < cyclic * lasu && lasu > 0)
    throw std::runtime_error("qcp_scan_cuda: bb.ndim==4 bb.size(1) < cyclic*lasu");
  if (tgt.dim() != 3) throw std::runtime_error("qcp_scan_cuda: tgt.dim() != 3)");
  if (bb.size(bb.dim() - 1) != 3) throw std::runtime_error("qcp_scan_cuda: bb.size(-1) != 3");
  if (tgt.size(tgt.dim() - 1) != 3) throw std::runtime_error("qcp_scan_cuda: tgt.size(-1) != 3)");
  if (ranges.size(1) != 2) throw std::runtime_error("qcp_scan_cuda: ranges.size(1) != 2");
  if (ranges.size(0) < 3)
    throw std::runtime_error("qcp_scan_cuda: ranges must have at least 3 rows to compute rms");
  if (tgt.size(0) != ranges.size(0) * cyclic)
    throw std::runtime_error("qcp_scan_cuda: tgt.size(0) != len(ranges)*cyclic");
  if (ranges.size(0) * cyclic != tgt.size(0)) throw std::runtime_error("tgt*cyclic len(ranges) must be same");
  if (bb.size(0) % cyclic) throw std::runtime_error("bb.size(0) must be a multiple of cyclic");
  if (ranges.size(0) > 7) throw std::runtime_error("qcp_scan_cuda: ranges.size(0) > 7 not allowed");
  return std::tuple(bb, tgt);
}

inline auto qcp_scan_get_idx_rms(Tensor ranges, uint64_t result) {
  Tensor idx = at::zeros({ranges.size(0)}, TensorOptions().device(kCUDA).dtype(kInt64));
  int index = ((result << 44) >> 44);
  float rms = ((float)(result >> 44)) / 10000.0;
  // cerr << result << " " << index << " " << rms << " " << allrms.min().item<float>() << endl;
  for (int j = 0; j < ranges.size(0); ++j) {
    int32_t cumprod = 1;
    for (int i = j + 1; i < ranges.size(0); ++i)
      cumprod *= (ranges[i][1] - ranges[i][0]).item<int>();
    idx[j] = ranges[j][0] + (index / cumprod) % (ranges[j][1] - ranges[j][0]).item<int>();
  }
  return std::tuple(idx, rms);
}
} // namespace qcpscan_impl

py::tuple qcp_scan_cuda(Tensor bb,
                        Tensor tgt,
                        Tensor ranges,
                        int cyclic,
                        int chainbreak,
                        int lasu,
                        int64_t const threads = 64,
                        bool rmsout = false) {
  using namespace qcpscan_impl;
  std::tie(bb, tgt) = qcp_scan_cuda_check_inputs(bb, tgt, ranges, cyclic, chainbreak, lasu);
  auto sizes = ranges.index({Slice(0, None), 1}) - ranges.index({Slice(0, None), 0});
  const int64_t ntotal = at::prod(sizes).item<int64_t>();
  // const int64_t threads = 64;
  const int64_t blocks((ntotal + threads - 1) / threads);
  Tensor tgtcen = tgt - tgt.mean(IntArrayRef({0, 1}));
  Tensor allrms = 9e9 * at::ones({rmsout ? ntotal : 0}, TensorOptions().device(kCUDA).dtype(kFloat));
  auto gpuresult = make_device_ptr<unsigned long long int>(1ull << 63);
  if (bb.dim() == 3) {
    qcp_scan_rmsd_cuda_kernel<float>
        <<<blocks, threads>>>(kmatxf3(bb), kmatxf3(tgtcen), kmatx2i(ranges), cyclic, chainbreak, lasu,
                              tgtcen.square().sum().item<float>(), ntotal, true,
                              rmsout ? allrms.data_ptr<float>() : nullptr, gpuresult);
  } else if (bb.dim() == 4) {
    qcp_scan_rmsd_cuda_kernel<float>
        <<<blocks, threads>>>(kten4f(bb), kmatxf3(tgtcen), kmatx2i(ranges), cyclic, chainbreak, lasu,
                              tgtcen.square().sum().item<float>(), ntotal, false,
                              rmsout ? allrms.data_ptr<float>() : nullptr, gpuresult);
  }
  finish_kernel();
  uint64_t result = device_ptr_copy_and_free(gpuresult);
  auto [idx, rms] = qcp_scan_get_idx_rms(ranges, result);
  if (rmsout)
    return py::make_tuple(idx.to(at::dtype<int64_t>()), allrms);
  else
    return py::make_tuple(idx.to(at::dtype<int64_t>()), rms);
}

__device__ inline float qcp_rmsd_impl_getlen(const float *__restrict__ len, int index) { return len[index]; }
__device__ inline double qcp_rmsd_impl_getlen(const double *__restrict__ len, int index) { return len[index]; }
__device__ inline float qcp_rmsd_impl_getlen(const float len, int) { return len; }
__device__ inline double qcp_rmsd_impl_getlen(const double len, int) { return len; }

template <class F, class Len, bool calcrot = false>
__global__ void qcp_calc_rmsd_cuda_kernel(const F *__restrict__ A,
                                          const F *__restrict__ E0,
                                          Len __restrict__ len,
                                          size_t ntotal,
                                          F *__restrict__ rmsd,
                                          F *__restrict__ rot = nullptr) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < ntotal) {
    rmsd[index] =
        compute_rms<F, calcrot>(&A[9 * index], E0[index], qcp_rmsd_impl_getlen(len, index), rot + 16 * index);
  }
}

auto qcp_calc_pre(Tensor iprod, Tensor E0, int threads) {
  assert(iprod.dim() == 3);
  assert(E0.dim() == 1);
  const int ntotal = iprod.size(0);
  const int blocks((ntotal + threads - 1) / threads);
  Tensor rmsd = at::zeros_like(E0, kF32);
  return std::tuple(rmsd, ntotal, blocks);
}

py::tuple qcp_calc_rmsd_cuda(Tensor iprod, Tensor E0, Tensor len, bool calcrot = false, int nthread = 128) {
  assert(len.dim() == 1);
  Tensor rmsd;
  int ntotal, blocks;
  std::tie(rmsd, ntotal, blocks) = qcp_calc_pre(iprod, E0, nthread);
  Tensor rot;
  if (calcrot) {
    rot = at::zeros({E0.size(0), 4, 4}, TensorOptions().dtype(E0.dtype()).device(kCUDA));
    AT_DISPATCH_FLOATING_TYPES(E0.type(), "qcp_calc_rmsd_cuda", ([&] {
                                 qcp_calc_rmsd_cuda_kernel<scalar_t, const scalar_t *, true>
                                     <<<blocks, nthread>>>(iprod.data_ptr<scalar_t>(), E0.data_ptr<scalar_t>(),
                                                           len.data_ptr<scalar_t>(), ntotal,
                                                           rmsd.data_ptr<scalar_t>(),
                                                           rot.data_ptr<scalar_t>());
                               }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(E0.type(), "qcp_calc_rmsd_cuda", ([&] {
                                 qcp_calc_rmsd_cuda_kernel<scalar_t, const scalar_t *, false>
                                     <<<blocks, nthread>>>(iprod.data_ptr<scalar_t>(), E0.data_ptr<scalar_t>(),
                                                           len.data_ptr<scalar_t>(), ntotal,
                                                           rmsd.data_ptr<scalar_t>());
                               }));
  }
  finish_kernel();
  return py::make_tuple(rmsd, rot);
}

py::tuple
qcp_calc_rmsd_cuda_fixlen(Tensor iprod, Tensor E0, double len, bool calcrot = false, int nthread = 128) {
  Tensor rmsd;
  int ntotal, blocks;
  std::tie(rmsd, ntotal, blocks) = qcp_calc_pre(iprod, E0, nthread);
  Tensor rot;
  if (calcrot) {
    rot = at::zeros({E0.size(0), 4, 4}, TensorOptions().dtype(E0.dtype()).device(kCUDA));
    AT_DISPATCH_FLOATING_TYPES(E0.type(), "qcp_calc_rmsd_cuda", ([&] {
                                 qcp_calc_rmsd_cuda_kernel<scalar_t, scalar_t, true><<<blocks, nthread>>>(
                                     iprod.data_ptr<scalar_t>(), E0.data_ptr<scalar_t>(), (scalar_t)len,
                                     ntotal, rmsd.data_ptr<scalar_t>(), rot.data_ptr<scalar_t>());
                               }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(E0.type(), "qcp_calc_rmsd_cuda", ([&] {
                                 qcp_calc_rmsd_cuda_kernel<scalar_t, scalar_t, false>
                                     <<<blocks, nthread>>>(iprod.data_ptr<scalar_t>(), E0.data_ptr<scalar_t>(),
                                                           (scalar_t)len, ntotal, rmsd.data_ptr<scalar_t>());
                               }));
  }
  finish_kernel();
  return py::make_tuple(rmsd, rot);
}

template <class F, bool calcrot>
__device__ inline F compute_rms(F const *__restrict__ A, F const E0, F const len, F *rot) {
  F C[4];

  F evecprec, evalprec;
  if constexpr (std::is_same<F, double>::value) {
    evecprec = 1e-5;
    evalprec = 1e-10;
  } else {
    evecprec = 1e-3;
    evalprec = 1e-6;
  }
  F Sxx = A[0], Sxy = A[1], Sxz = A[2], Syx = A[3], Syy = A[4], Syz = A[5], Szx = A[6], Szy = A[7], Szz = A[8];
  F Sxx2 = Sxx * Sxx, Syy2 = Syy * Syy, Szz2 = Szz * Szz, Sxy2 = Sxy * Sxy, Syz2 = Syz * Syz, Sxz2 = Sxz * Sxz,
    Syx2 = Syx * Syx, Szy2 = Szy * Szy, Szx2 = Szx * Szx;
  F SyzSzymSyySzz2 = 2.0 * (Syz * Szy - Syy * Szz);
  F Sxx2Syy2Szz2Syz2Szy2 = Syy2 + Szz2 - Sxx2 + Syz2 + Szy2;
  C[2] = -2.0 * (Sxx2 + Syy2 + Szz2 + Sxy2 + Syx2 + Sxz2 + Szx2 + Syz2 + Szy2);
  C[1] = 8.0 * (Sxx * Syz * Szy + Syy * Szx * Sxz + Szz * Sxy * Syx - Sxx * Syy * Szz - Syz * Szx * Sxy -
                Szy * Syx * Sxz);
  F SxzpSzx = Sxz + Szx, SyzpSzy = Syz + Szy, SxypSyx = Sxy + Syx, SyzmSzy = Syz - Szy, SxzmSzx = Sxz - Szx,
    SxymSyx = Sxy - Syx, SxxpSyy = Sxx + Syy, SxxmSyy = Sxx - Syy;
  F Sxy2Sxz2Syx2Szx2 = Sxy2 + Sxz2 - Syx2 - Szx2;
  C[0] = Sxy2Sxz2Syx2Szx2 * Sxy2Sxz2Syx2Szx2 +
         (Sxx2Syy2Szz2Syz2Szy2 + SyzSzymSyySzz2) * (Sxx2Syy2Szz2Syz2Szy2 - SyzSzymSyySzz2) +
         (-(SxzpSzx) * (SyzmSzy) + (SxymSyx) * (SxxmSyy - Szz)) *
             (-(SxzmSzx) * (SyzpSzy) + (SxymSyx) * (SxxmSyy + Szz)) +
         (-(SxzpSzx) * (SyzpSzy) - (SxypSyx) * (SxxpSyy - Szz)) *
             (-(SxzmSzx) * (SyzmSzy) - (SxypSyx) * (SxxpSyy + Szz)) +
         (+(SxypSyx) * (SyzpSzy) + (SxzpSzx) * (SxxmSyy + Szz)) *
             (-(SxymSyx) * (SyzmSzy) + (SxzpSzx) * (SxxpSyy + Szz)) +
         (+(SxypSyx) * (SyzmSzy) + (SxzmSzx) * (SxxmSyy - Szz)) *
             (-(SxymSyx) * (SyzpSzy) + (SxzmSzx) * (SxxpSyy - Szz));

  // Newton-Raphson
  F mxEigenV = E0;
  for (int i = 0; i < 50; ++i) {
    F oldg = mxEigenV;
    F x2 = mxEigenV * mxEigenV;
    F b = (x2 + C[2]) * mxEigenV;
    F a = b + C[1];
    F delta = ((a * mxEigenV + C[0]) / (2.0 * x2 * mxEigenV + b + a));
    mxEigenV -= delta;
    // printf("\n diff[%3d]: %16g %16g %16g", i, mxEigenV - oldg,
    // evalprec*mxEigenV, mxEigenV);
    if (fabs(mxEigenV - oldg) < fabs(evalprec * mxEigenV)) break;
  }
  F rms = sqrt(fabs((F)2.0 * (E0 - mxEigenV) / len));
  if constexpr (!calcrot) {

    return rms;

  } else {

    F a11 = SxxpSyy + Szz - mxEigenV;
    F a12 = SyzmSzy;
    F a13 = -SxzmSzx;
    F a14 = SxymSyx;
    F a21 = SyzmSzy;
    F a22 = SxxmSyy - Szz - mxEigenV;
    F a23 = SxypSyx;
    F a24 = SxzpSzx;
    F a31 = a13;
    F a32 = a23;
    F a33 = Syy - Sxx - Szz - mxEigenV;
    F a34 = SyzpSzy;
    F a41 = a14;
    F a42 = a24;
    F a43 = a34;
    F a44 = Szz - SxxpSyy - mxEigenV;
    F a3344_4334 = a33 * a44 - a43 * a34;
    F a3244_4234 = a32 * a44 - a42 * a34;
    F a3243_4233 = a32 * a43 - a42 * a33;
    F a3143_4133 = a31 * a43 - a41 * a33;
    F a3144_4134 = a31 * a44 - a41 * a34;
    F a3142_4132 = a31 * a42 - a41 * a32;
    F q1 = a22 * a3344_4334 - a23 * a3244_4234 + a24 * a3243_4233;
    F q2 = -a21 * a3344_4334 + a23 * a3144_4134 - a24 * a3143_4133;
    F q3 = a21 * a3244_4234 - a22 * a3144_4134 + a24 * a3142_4132;
    F q4 = -a21 * a3243_4233 + a22 * a3143_4133 - a23 * a3142_4132;

    F qsqr = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4;

    // The following code tries to calculate another column in the adjoint matrix
    //   when the norm of the current column is too small. Usually this block will
    //   never be activated.  To be absolutely safe this should be uncommented,
    //   but it is most likely unnecessary.

    if (qsqr < evecprec) {
      q1 = a12 * a3344_4334 - a13 * a3244_4234 + a14 * a3243_4233;
      q2 = -a11 * a3344_4334 + a13 * a3144_4134 - a14 * a3143_4133;
      q3 = a11 * a3244_4234 - a12 * a3144_4134 + a14 * a3142_4132;
      q4 = -a11 * a3243_4233 + a12 * a3143_4133 - a13 * a3142_4132;
      qsqr = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4;

      if (qsqr < evecprec) {
        F a1324_1423 = a13 * a24 - a14 * a23, a1224_1422 = a12 * a24 - a14 * a22;
        F a1223_1322 = a12 * a23 - a13 * a22, a1124_1421 = a11 * a24 - a14 * a21;
        F a1123_1321 = a11 * a23 - a13 * a21, a1122_1221 = a11 * a22 - a12 * a21;

        q1 = a42 * a1324_1423 - a43 * a1224_1422 + a44 * a1223_1322;
        q2 = -a41 * a1324_1423 + a43 * a1124_1421 - a44 * a1123_1321;
        q3 = a41 * a1224_1422 - a42 * a1124_1421 + a44 * a1122_1221;
        q4 = -a41 * a1223_1322 + a42 * a1123_1321 - a43 * a1122_1221;
        qsqr = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4;

        if (qsqr < evecprec) {
          q1 = a32 * a1324_1423 - a33 * a1224_1422 + a34 * a1223_1322;
          q2 = -a31 * a1324_1423 + a33 * a1124_1421 - a34 * a1123_1321;
          q3 = a31 * a1224_1422 - a32 * a1124_1421 + a34 * a1122_1221;
          q4 = -a31 * a1223_1322 + a32 * a1123_1321 - a33 * a1122_1221;
          qsqr = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4;

          if (qsqr < evecprec) {
            // if qsqr is still too small, return the identity matrix.
            rot[0] = rot[5] = rot[10] = 1.0;
            rot[1] = rot[2] = rot[4] = rot[6] = rot[8] = rot[9] = 0.0;
            return rms;
          }
        }
      }
    }

    F normq = sqrt(qsqr);
    q1 /= normq;
    q2 /= normq;
    q3 /= normq;
    q4 /= normq;

    F a2 = q1 * q1;
    F x2 = q2 * q2;
    F y2 = q3 * q3;
    F z2 = q4 * q4;

    F xy = q2 * q3;
    F az = q1 * q4;
    F zx = q4 * q2;
    F ay = q1 * q3;
    F yz = q3 * q4;
    F ax = q1 * q2;

    rot[0] = a2 + x2 - y2 - z2;
    rot[4] = 2 * (xy + az);
    rot[8] = 2 * (zx - ay);
    rot[1] = 2 * (xy - az);
    rot[5] = a2 - x2 + y2 - z2;
    rot[9] = 2 * (yz + ax);
    rot[2] = 2 * (zx + ay);
    rot[6] = 2 * (yz - ax);
    rot[10] = a2 - x2 - y2 + z2;

    return rms;
  }
}

} // namespace qcprms
} // namespace ipd
