#include <ipd_util.hpp>

namespace ipd {
namespace voxel {

Tensor eval_func(Tensor x, std::string func, Tensor funcarg);

py::tuple create_voxel_grid(
    Tensor xyz, float resl, std::string func, Tensor funcarg, Tensor nthread, Tensor repulsive_onl);

Tensor score_voxel_grid(Tensor vox,
                        Tensor voxposinv,
                        Tensor xyz,
                        Tensor xyzpos,
                        Tensor lb,
                        float resl,
                        Tensor nthread,
                        Tensor repulsive_only,
                        Tensor symx,
                        float symclashdist);

PYBIND11_MODULE(_voxel, m) {
  m.def("eval_func", &eval_func, "x"_a, "func"_a, "funcarg"_a);
  m.def("create_voxel_grid", &create_voxel_grid, "xyz"_a, "resl"_a, "func"_a, "funcarg"_a, "nthread"_a,
        "repulsive_only"_a = at::empty(0, at::dtype(kBool)));
  m.def("score_voxel_grid", &score_voxel_grid, "vox"_a, "voxposinv"_a, "xyz"_a, "xyzpos"_a, "lb"_a, "resl"_a,
        "nthread"_a, "repulsive_only"_a = at::empty(0, at::dtype(kBool)), "symx"_a = at::empty(0),
        "symclashdist"_a = 0.0);
}

} // namespace voxel
} // namespace ipd
