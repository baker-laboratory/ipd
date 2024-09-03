#include <_shapes.hpp>
#include <ipd_util.hpp>

namespace ipd {
namespace sampling {

py::tuple tip_atom_placer(
    Tensor nblist, Tensor don, Tensor acc, Tensor tipxyz, Tensor tipdon, Tensor tipacc, py::dict params);

Tensor rand_xform(int64_t n,
                  Tensor cart_mean,
                  Tensor cart_sd,
                  bool cart_uniform,
                  float quat_height,
                  bool orinormal,
                  string dtype,
                  int nthread,
                  int64_t seed,
                  string gentype);

Tensor sort_inplace_topk_float(Tensor data, int k);

py::tuple welzl_bounding_sphere_tensor(Tensor xyz) {
  if (xyz.size(1) != 3 || xyz.dim() != 2) throw std::runtime_error("xyz must be Nx3 tensor");
  xyz = xyz.to(kF32).to(kCPU);
  auto sph = welzl_bounding_sphere(kvecx3f(xyz));
  Tensor cen = at::zeros({3}, TensorOptions().device(xyz.device()).dtype(xyz.dtype()));
  cen[0] = sph.cen[0];
  cen[1] = sph.cen[1];
  cen[2] = sph.cen[2];
  return py::make_tuple(cen, sph.rad);
}

PYBIND11_MODULE(_sampling, m) {
  m.def("tip_atom_placer", &tip_atom_placer, "nblist"_a, "don"_a, "acc"_a, "tipxyz"_a, "tipdon"_a, "tipacc"_a,
        "params"_a);

  m.def("rand_xform", &rand_xform, "n"_a, "cart_mean"_a, "cart_sd"_a = 1, "cart_uniform"_a = false,
        "quat_height"_a = 0, "orinormal"_a = false, "dtype"_a = "torch.float32", "nthread"_a = 256,
        "seed"_a = std::numeric_limits<int64_t>::max(), "gentype"_a = "curandState");
  m.def("sort_inplace_topk_float", &sort_inplace_topk_float, "data"_a, "k"_a);
  m.def("welzl_bounding_sphere", &welzl_bounding_sphere_tensor, "xyz"_a);
}

} // namespace sampling
} // namespace ipd
