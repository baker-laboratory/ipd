#include <ipd_cuda_util.hpp>

namespace ipd {
namespace voxel {

template <typename F> struct Clash {
  __device__ static F call(kMatX1f arg, F dist) {
    if (dist > arg(1)) return 0.0;
    if (dist < arg(0)) return 1.0;
    return (arg(1) - dist) / (arg(1) - arg(0));
  }
};
template <typename F> struct Contact {
  __device__ static F call(kMatX1f arg, F dist) {
    // 0 CLEND CTBEG CTEND END
    F cl = arg[0];
    F ct = arg[1];
    F clend = arg[2];
    F ctbeg = arg[3];
    F ctend = arg[4];
    F end = arg[5];
    if (dist < clend) return cl;
    if (dist < ctbeg) return cl + (ct - cl) * (dist - clend) / (ctbeg - clend);
    if (dist < ctend) return ct;
    if (dist < end) return ct * (end - dist) / (end - ctend);
    return 0;
  }
};

template <typename F, template <typename> typename Func>
__global__ void eval_func_kernel(kMatX1<F> dist, kMatX1<F> arg, kMatX1<F> out) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < dist.rows()) out(i) = Func<F>::call(arg, dist(i));
}

Tensor eval_func(Tensor dist, std::string func, Tensor funcarg) {
  CHECK_INPUT(dist)
  CHECK_INPUT(funcarg)
  Tensor out = at::zeros_like(dist);
  if (func == "clash") {
    eval_func_kernel<float, Clash><<<dist.size(0), 128>>>(kmatx1f(dist), kmatx1f(funcarg), kmatx1f(out));
  } else if (func == "contact") {
    eval_func_kernel<float, Contact><<<dist.size(0), 128>>>(kmatx1f(dist), kmatx1f(funcarg), kmatx1f(out));
  } else {
    throw std::runtime_error("Unknown function " + func);
  }
  return out;
}

template <typename F, template <typename> typename Func>
__global__ void create_voxel_grid_kernel(
    kMatX3<F> xyz, kMat3<F> lb, int irad, kMatX1<F> funcarg, F resl, kTen3h vox, kMatX1<bool> repulsive_only) {
  auto [ix, iy, iz, ok] = kernel_index3d<int>(dim3(xyz.rows(), 2 * irad + 1, 2 * irad + 1));
  if (!ok) return;
  Vec3<F> cen = xyz.row(ix);
  Vec3<int> icen = ((cen - lb) / resl).template cast<int>();
  int i = iy - irad - 1 + icen(0);
  int j = iz - irad - 1 + icen(1);
  for (int k = icen(2) - irad; k <= icen(2) + irad; ++k) {
    Vec3<F> bincen = lb + Vec3<F>(i, j, k) * resl;
    F val = Func<F>::call(funcarg, (bincen - cen).norm());
    if (repulsive_only.rows() && repulsive_only[ix]) val = 3 * fabs(val);
    if (val != 0) vox(i, j, k) += val;
  }
}

py::tuple create_voxel_grid(
    Tensor xyz, float resl, std::string func, Tensor funcarg, Tensor nthread, Tensor repulsive_only) {
  CHECK_INPUT(xyz)
  CHECK_INPUT(funcarg)
  if (xyz.dim() != 2) throw std::runtime_error("xyz.dim() != 2");
  if (xyz.size(1) != 3) throw std::runtime_error("xyz.size(1) != 3");
  xyz = xyz.to(kFloat32);
  Tensor lb = get<0>(xyz.min(0)) - funcarg[-1] - resl;
  Tensor ub = get<0>(xyz.max(0)) + funcarg[-1] + resl;
  funcarg = funcarg.to(kFloat32);
  int irad = ceil(funcarg[-1].item<float>() / resl);
  Tensor _ncell = ceil((ub - lb) / resl).cpu().to(kInt64);
  auto voxsize = c10::IntArrayRef(_ncell.data_ptr<int64_t>(), 3);
  Tensor vox = at::zeros(voxsize, kF16).to(at::device(kCUDA));
  auto [thread, block, size] = blocks3d(Vec3<int>(xyz.size(0), irad * 2 + 1, irad * 2 + 1), nthread);
  if (func == "clash") {
    assert(funcarg.size(0) == 2);
    create_voxel_grid_kernel<float, Clash><<<block, thread>>>(kmatx3f(xyz), kmat3f(lb), irad, kmatx1f(funcarg),
                                                              resl, kten3h(vox), kmatx1b(repulsive_only));
  } else if (func == "contact") {
    assert(funcarg.size(0) == 6);
    create_voxel_grid_kernel<float, Contact><<<block, thread>>>(
        kmatx3f(xyz), kmat3f(lb), irad, kmatx1f(funcarg), resl, kten3h(vox), kmatx1b(repulsive_only));
  } else {
    throw std::runtime_error("Unknown function " + func);
  }
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  return py::make_tuple(vox, lb);
}

template <typename F, typename M, typename P> __device__ Vec3<F> xform_point3(M xform, P point) {
  Vec4<F> pt(point[0], point[1], point[2], 1);
  Vec4<F> p = xform * pt;
  Vec3<F> xformed(p[0], p[1], p[2]);
  return xformed;
}

template <typename F>
__global__ void score_voxel_grid_kernel(kTen3h vox,
                                        kTen3<F> voxposinv,
                                        kMatX3<F> xyz,
                                        kTen3<F> xyzpos,
                                        Vec3f lb,
                                        F resl,
                                        kTen2<F> score,
                                        kMat44<F> symx,
                                        kMatX1<bool> repulsive_only,
                                        F symclashdist = 0) {
  // int nxyz = ceil((F)xyz.rows() / (F)xyzchunk);
  // auto [ixyz0, ixyzpos, ok] = kernel_index2d<int>(dim3(nxyz, xyzpos.dimension(0), 1));
  auto [ivoxpos, ixyzpos, ok] = kernel_index2d<int>(voxposinv.dimension(0), xyzpos.dimension(0));
  if (!ok) return;
  kMat44<F> tovox(&voxposinv(ivoxpos, 0, 0));
  kMat44<F> xyz2global(&xyzpos(ixyzpos, 0, 0));
  Mat44<F> xform = tovox * xyz2global;
  // kprint("symx ", symx, "\n");
  // kprint("tovox ", tovox, "\n");
  // kprint("xyz2global ", xyz2global, "\n");
  // kprint("xform ", xform, "\n");
  // kprint("symx ", symx, "\n");
  for (int ixyz = 0; ixyz < xyz.rows(); ++ixyz) {
    // printf("ixyz %i\n", ixyz);
    // if (ixyz >= xyz.rows()) break;
    Vec3<F> xyzvox = xform_point3<F>(xform, xyz.row(ixyz));
    // kprint("xyzvox ", xyzvox, "\n");
    Ary3<int> ivox = ((xyzvox - lb) / resl).template cast<int>();
    if ((ivox < 0).any()) continue;
    bool out_of_bound = false;
    for (int i = 0; i < 3; ++i)
      if (ivox[i] >= vox.dimension(i)) out_of_bound = true;
    if (!out_of_bound) {
      F delta = vox(ivox[0], ivox[1], ivox[2]);
      if (repulsive_only.rows() && repulsive_only[ixyz]) delta = 3 * fabs(delta);
      if (delta != 0) score(ivoxpos, ixyzpos) += delta;
    }
  }
  if (symclashdist > 0) {
    Mat44<F> symxform = symx * xyz2global;
    // printf("xyz.rows() %i\n", xyz.rows());
    for (int ixyz = 0; ixyz < xyz.rows(); ++ixyz) {
      Vec3<F> xyzglobal = xform_point3<F>(xyz2global, xyz.row(ixyz));
      for (int jxyz = 0; jxyz < xyz.rows(); ++jxyz) {
        Vec3<F> sympt = xform_point3<F>(symxform, xyz.row(jxyz));
        // printf("%i %i dist %f\n", ixyz, jxyz, (xyzglobal - sympt).norm());
        // kprint("xyzlocal ", Vec3<F>(xyz.row(ixyz)), "\n");
        // kprint("xyzglobal ", xyzglobal, "\n");
        // kprint("sympt ", sympt, "\n");
        if ((xyzglobal - sympt).squaredNorm() < symclashdist * symclashdist) {
          score(ivoxpos, ixyzpos) += 9e9;
          goto done;
        }
      }
    }
  }
done:;
}

// template <typename F>
// __global__ void score_voxel_grid_kernel_rt(
//     kTen3h vox, kMatX3<F> xyz, kTen3f rot, kMatX3<F> trans, Vec3f lb, F resl, kVecX<F> score, int
//     xyzchunk)
//     {
//   int nxyz = ceil((F)xyz.rows() / (F)xyzchunk);
//   // auto [ixyz0, ixyzpos, ok] = kernel_index2d<int>(dim3(nxyz, rot.dimension(0), 1));
//   auto [ixyzpos, ok] = kernel_index1d<int>(rot.dimension(0));
//   if (!ok) return;
//   // ixyz0 *= xyzchunk;
//   kMat33<F> R(&rot(ixyzpos, 0, 0));
//   Vec3<F> T(trans.row(ixyzpos));
//   // for (int ixyz = ixyz0; ixyz < ixyz0 + xyzchunk; ++ixyz) {
//   for (int ixyz = 0; ixyz < xyz.rows(); ++ixyz) {
//     if (ixyz >= xyz.rows()) break;
//     Vec3<F> placed = R * xyz.row(ixyz).transpose() + T;
//     Ary3<int> ivox = ((placed - lb) / resl).template cast<int>();
//     if ((ivox < 0).any()) continue;
//     bool out_of_bound = false;
//     for (int i = 0; i < 3; ++i)
//       if (ivox[i] >= vox.dimension(i)) out_of_bound = true;
//     if (out_of_bound) continue;
//     score[ixyzpos] += vox(ivox[0], ivox[1], ivox[2]);
//   }
// }

Tensor score_voxel_grid(Tensor vox,
                        Tensor voxposinv,
                        Tensor xyz,
                        Tensor xyzpos,
                        Tensor lb,
                        float resl,
                        Tensor nthread,
                        Tensor repulsive_only,
                        Tensor symx,
                        float symclashdist) {
  CHECK_INPUT(vox)
  CHECK_INPUT(voxposinv)
  CHECK_INPUT(xyz)
  CHECK_INPUT(xyzpos)
  CHECK_INPUT(lb)
  CHECK_INPUT(symx)
  if (vox.dim() != 3) throw std::runtime_error("vox.dim() != 3)");
  if (voxposinv.dim() != 3) throw std::runtime_error("voxposinv.dim() != 3");
  if (voxposinv.size(1) != 4) throw std::runtime_error("voxposinv.size(1) != 4");
  if (voxposinv.size(2) != 4) throw std::runtime_error("voxposinv.size(2) != 4");
  if (xyz.dim() != 2) throw std::runtime_error("xyz.dim() != 2");
  if (xyz.size(1) != 3) throw std::runtime_error("xyz.size(1) != 3");
  if (xyzpos.dim() != 3) throw std::runtime_error("xyzpos.dim() != 3");
  if (xyzpos.size(1) != 4) throw std::runtime_error("xyzpos.size(1) != 4");
  if (xyzpos.size(2) != 4) throw std::runtime_error("xyzpos.size(2) != 4");
  if (symx.size(0)) {
    if (symx.dim() != 2) throw std::runtime_error("symx.dim() != 2");
    if (symx.size(0) != 4) throw std::runtime_error("symx.size(0) != 4");
    if (symx.size(1) != 4) throw std::runtime_error("symx.size(1) != 4");
  }
  xyz = xyz.to(at::dtype(kF32));
  xyzpos = xyzpos.to(at::dtype(kF32));
  lb = lb.to(at::dtype(kF32));
  // int xyzchunk = 999999; // nthread[2].item<int>();
  // int xyzsize = ceil((float)xyz.size(0) / (float)xyzchunk);
  auto [thread, block, size] = blocks2d(voxposinv.size(0), xyzpos.size(0), nthread);
  // cerr << "thread.. " << thread << endl;
  // cerr << "block... " << block << endl;
  // cerr << "size.... " << size << endl;
  Tensor score = at::zeros({voxposinv.size(0), xyzpos.size(0)}, xyzpos.options());
  score_voxel_grid_kernel<<<block, thread>>>(kten3h(vox), kten3f(voxposinv), kmatx3f(xyz), kten3f(xyzpos),
                                             vec3f(lb), resl, kten2f(score), kmat44f(symx),
                                             kmatx1b(repulsive_only), symclashdist);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  return score;
}

} // namespace voxel
} // namespace ipd
