import pytest

pytest.importorskip('torch')
import os

import torch as th  # type: ignore
from numba import cuda

import ipd as ipd

def main():

    # Should also be True
    print(cuda.is_float16_supported())

    return

    os.environ['NUMBA_ENABLE_CUDASIM'] = '0'
    t = ipd.dev.Timer()
    s = 0
    for i in range(10000):
        # N = 8_388_608 +0
        N = int(1e6)
        # print(N/ 2**23)
        t.checkpoint('none')
        x1 = ipd.samp.randxform(N)
        # x2 = ipd.samp.randxform(N)
        # x1 = ipd.h.rand(N, device='cuda')
        # x2 = ipd.h.rand(N, device='cuda')
        s += x1[-1, -1, -1]  #+ x2[-1,-1,-1]
        t.checkpoint('randxform')
        continue
        x3 = th.matmul(x1, x2)
        s += x3[-1, -1, -1]
        t.checkpoint('matmul')
    print(min(t.checkpoints['randxform']) * 1000)

# @cuda.jit(device=True)
# def add3(a,b)
#     return (a[0]+b[0],a[1]+b[1],a[2]+b[2])
# @cuda.jit(device=True)
# def sub3(a,b)
#     return (a[0]-b[0],a[1]-b[1],a[2]-b[2])
# @cuda.jit(device=True)
# def dot3(a,b)
#     return (a[0]*b[0]+a[1]*b[1]+a[2]*b[2])
# @cuda.jit(device=True)
# def mul3(a,b)
#     return (a*b[0],a*b[1],a*b[2])
# @cuda.jit(device=True)
# def norm3(a)
#     return math.sqrt(a[0]*b[0]+a[1]*b[1]+a[2]*b[2])

# @cuda.jit(device=True)
# def makesph():
#     return (0.0,0.0,0.0,1)

# # @cuda.jit(device=True)
#   # Sphere(Vec3 c, F r) : cen(c), rad(r) {}
# @cuda.jit(device=True)
# def makesph(p1):
#     return (0.0,0.0,0.0,0.0)

# @cuda.jit(device=True)
# def makesph(O, A):
#   # Sphere(Vec3 O, Vec3 A) {
#     a = sub3(A,O)
#     o = mul3(0.5,a)
#     rad = norm3(o)+0.000001
#     # cen = n
#     # Vec3 a = A - O;
#     # Vec3 o = 0.5 * a;
#     # rad = o.norm() + epsilon2<F>();
#     cen = add3(O, o)
#     return (*cen,rad)

# @cuda.jit(device=True)
# def makesph(O,A,B):
#   # Sphere(Vec3 O, Vec3 A, Vec3 B) {
#     a = sub3(A, O)
#     b = sub3(B, O)
#     det_2 = 2.0 * ((a.cross(b)).dot(a.cross(b)));
#     Vec3 o = (b.dot(b) * ((a.cross(b)).cross(a)) +
#               a.dot(a) * (b.cross(a.cross(b)))) /
#              det_2;
#     rad = o.norm() + epsilon2<F>();
#     cen = O + o;
#   }
# @cuda.jit(device=True)
# def makesph(O,A,B,C):
#   Sphere(Vec3 O, Vec3 A, Vec3 B, Vec3 C) {
#     Vec3 a = A - O, b = B - O, c = C - O;
#     Mat3 cols;
#     cols.col(0) = a;
#     cols.col(1) = b;
#     cols.col(2) = c;
#     F det_2 = 2.0 * Mat3(cols).determinant();
#     Vec3 o = (c.dot(c) * a.cross(b) + b.dot(b) * c.cross(a) +
#               a.dot(a) * b.cross(c)) /
#              det_2;
#     rad = o.norm() + epsilon2<F>();
#     cen = O + o;
#     pass

# @cuda.jit('f4[:],f4[:,:],f4,f4[:,:],f4', device=True)
# def welzl_bounding_sphere_impl(out, pts, idx, sos,          nsos):
#   // if no input points, the recursion has bottomed out. Now compute an
#   // exact sphere based on points in set of support (zero through four points)
#   if (index == 0):
#     match numsos:
#       case 0:
#         return makesph(out);
#       case 1:
#         return makesph(out,sos[0]);
#       case 2:
#         return makesph(out,sos[0], sos[1]);
#       case 3:
#         return makesph(out,sos[0], sos[1], sos[2]);
#       case 4:
#         return makesph(out,sos[0], sos[1], sos[2], sos[3]);
#   idx -= 1
#   Sph smallestSphere =
#       welzl_bounding_sphere_impl(points, index, sos, numsos);  // (*)

#   if (smallestSphere.contains(points[index])) return smallestSphere;

#   if (numsos == 4):
#     return smallestSphere;
#   for i in range(3): sos[numsos,i] = pts[index,i];
#   return welzl_bounding_sphere_impl(points, index, sos, numsos + 1);

# template <class Ary, class Sph, bool range>
# struct UpdateBounds {
#   static void update_bounds(Ary const& points, Sph& sph) {}
# };

# template <class Ary, class Sph>
# struct UpdateBounds<Ary, Sph, false> {
#   static void update_bounds(Ary const& points, Sph& sph) {}
# };
# template <class Ary, class Sph>
# struct UpdateBounds<Ary, Sph, true> {
#   static void update_bounds(Ary const& points, Sph& sph) {
#     sph.lb = 2000000000;
#     sph.ub = -2000000000;
#     for (size_t i = 0; i < points.size(); ++i) {
#       sph.lb = std::min(sph.lb, points.get_index(i));
#       sph.ub = std::max(sph.ub, points.get_index(i));
#     }
#   }
# };

# template <bool range = false, class Ary>
# auto welzl_bounding_sphere(Ary const& points) noexcept {
#   using Pt = typename Ary::value_type;
#   using Sph = Sphere<typename Pt::Scalar>;
#   std::vector<Pt> sos(4);
#   Sph bound = welzl_bounding_sphere_impl(points, points.size(), sos, 0);
#   UpdateBounds<Ary, Sph, range>::update_bounds(points, bound);
#   // if (bound.lb < 0 || bound.lb > bound.ub || bound.ub > 2000)
#   // std::cout << bound.lb << " " << bound.ub << std::endl;
#   return bound;
# }

if __name__ == '__main__':
    main()
