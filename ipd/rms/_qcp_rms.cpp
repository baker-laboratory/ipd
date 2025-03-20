#include <ipd_util.hpp>

namespace ipd {
namespace qcprms {

py::tuple
qcp_calc_rmsd_cuda_fixlen(Tensor iprod, Tensor E0, double len, bool getrot = false, int nthread = 128);
py::tuple qcp_calc_rmsd_cuda(Tensor iprod, Tensor E0, Tensor len, bool getrot = false, int nthread = 128);

py::tuple qcp_scan_cuda(Tensor xyz1,
                        Tensor xyz2,
                        Tensor lbub,
                        int cyclic,
                        int chainbreak,
                        int lasu,
                        int64_t const nthread,
                        bool rmsout);

template <typename F>
int qcp_calc_rmsd_maybe_rotation(F *rot, F *A, F *rmsd, F E0, F len, F minScore, int rotsize = 4);

template <typename F>
F qcp_rmsd_impl(Tensor xyz1_in,
                Tensor xyz2_in,
                F *rot = nullptr,
                F *cen1 = nullptr,
                F *cen2 = nullptr,
                bool showme = false) {

  if (xyz1_in.sizes()[0] != xyz2_in.sizes()[0]) throw std::runtime_error("xyz1 and xyz2 not same size");
  assert(xyz1_in.dim() == 2);
  assert(xyz2_in.dim() == 2);

  Tensor xyz1 = xyz1_in.clone();
  Tensor xyz2 = xyz2_in.clone();
  Tensor _cen1 = xyz1.mean(0);
  Tensor _cen2 = xyz2.mean(0);
  xyz1 -= _cen1;
  xyz2 -= _cen2;

  Tensor iprod = linalg_matmul(transpose(xyz1, 1, 0), xyz2);
  F E0 = (xyz1.square().sum() + xyz2.square().sum()).item<F>() / 2.0;

  // cout << iprod << " " << E0 << endl;

  if (showme) {
    cout << "REF CEN1 " << xyz1_in.mean(0) << endl;
    cout << "REF CEN2 " << xyz2_in.mean(0) << endl;
    cout << "REF IPROD" << endl;
    cout << iprod << endl;
    cout << "REF sqnorm1 " << xyz1.square().sum() << endl;
    cout << "REF sqnorm2 " << xyz2.square().sum() << endl;
    cout << "REF E0 " << E0 << " xyz1 " << xyz1 << endl;
  }

  // cout << "SNG " << (((F *)iprod.data_ptr<F>())[0]) << " " << E0 << " "
  // << F(xyz1.size(0)) << endl;

  F rmsd;
  qcp_calc_rmsd_maybe_rotation<F>((F *)rot, iprod.data_ptr<F>(), &rmsd, E0, F(xyz1.size(0)), F(-1.0), 3);

  if (cen1 != nullptr)
    for (int i = 0; i < 3; ++i) {
      cen1[i] = _cen1.accessor<F, 1>()[i];
    }
  if (cen2 != nullptr)
    for (int i = 0; i < 3; ++i)
      cen2[i] = _cen2.accessor<F, 1>()[i];

  return rmsd;
}

template <typename F> F qcp_rmsd(Tensor xyz1, Tensor xyz2) {
  F c1[3], c2[3];
  return qcp_rmsd_impl<F>(xyz1, xyz2, (F *)nullptr, c1, c2);
}

template <typename F> py::tuple qcp_rmsd_align(Tensor xyz1, Tensor xyz2) {
  Tensor R = at::zeros(IntArrayRef({3, 3}), at::dtype<F>());
  Tensor T = at::zeros(IntArrayRef({3}), at::dtype<F>());
  Tensor c1 = at::zeros(IntArrayRef({3}), at::dtype<F>());
  Tensor c2 = at::zeros(IntArrayRef({3}), at::dtype<F>());
  F rms;
  {
    // py::gil_scoped_release release;
    rms = qcp_rmsd_impl<F>(xyz1, xyz2, (F *)R.data_ptr<F>(), (F *)c1.data_ptr<F>(), (F *)c2.data_ptr<F>());
    T = at::matmul(R, -c1) + c2;
  }
  return py::make_tuple(rms, R, T);
}

template <typename F>
std::tuple<Tensor, Tensor> qcp_rmsd_raw_vec(Tensor iprod, Tensor E0, Tensor len, bool getrot) {
  iprod = iprod.cpu();
  E0 = E0.cpu();
  len = len.cpu();
  auto shape = E0.sizes();
  iprod = iprod.view({-1, 3, 3});
  E0 = E0.view({-1});
  len = len.view({-1});
  Tensor out = at::zeros_like(E0);
  Tensor rot = at::zeros({(getrot ? (int)len.size(0) : 0), 4, 4}, at::dtype<F>());
  {
    // py::gil_scoped_release release;
    for (int i = 0; i < len.size(0); ++i) {
      if (getrot)
        qcp_calc_rmsd_maybe_rotation<F>(rot[i].data_ptr<F>(), iprod[i].data_ptr<F>(), out[i].data_ptr<F>(),
                                        E0[i].item<F>(), len[i].item<F>(), (F)-1.0);
      else
        qcp_calc_rmsd_maybe_rotation<F>((F *)nullptr, iprod[i].data_ptr<F>(), out[i].data_ptr<F>(),
                                        E0[i].item<F>(), len[i].item<F>(), (F)-1.0);
    }
  }
  return std::make_tuple(out.view(shape), rot);
}

// template <typename F> Tensor qcp_rmsd_raw_vec(Tensor iprod, Tensor E0, F len)
// { Tensor l = at::ones_like(E0) * len; return qcp_rmsd_raw_vec(iprod, E0,
// len);
// }

template <typename F> py::object qcp_rmsd_vec(Tensor a_in, Tensor b_in, bool getrot) {
  auto M = a_in.size(0);
  auto N = b_in.size(0);
  // Tensor out = at::zeros(IntArrayRef({M, N}), at::dtype<F>()).view({-1});
  Tensor rms, xform;
  {
    // py::gil_scoped_release release;
    Tensor a = a_in.clone();
    Tensor b = b_in.clone();
    if (a.dim() != 3 || b.dim() != 3) throw std::runtime_error("ndim must be 3");
    if (a.size(1) != b.size(1)) throw std::runtime_error("mismatched shape[1]");
    Tensor c1 = a.mean(1);
    Tensor c2 = b.mean(1);
    a -= c1.unsqueeze(1);
    b -= c2.unsqueeze(1);
    auto a2 = transpose(a.unsqueeze(1), -1, -2);
    auto b2 = b.unsqueeze(0);
    Tensor iprod = at::matmul(transpose(a.unsqueeze(1), -1, -2), b.unsqueeze(0)).view({-1, 3, 3});
    Tensor E0 = (a2.square().sum({2, 3}) + b2.square().sum({2, 3})).view({-1}) / 2.0;
    // for (int i = 0; i < E0.size(0); ++i) {
    // F rmsd;
    // qcp_calc_rmsd_maybe_rotation((F *)nullptr, (F *)(iprod[i].data_ptr<F>()),
    // &rmsd, E0[i].item<F>(), F(a.size(1)),
    // F(-1.0));
    // out[i] = rmsd;
    // }
    Tensor len = at::ones_like(E0) * F(a_in.size(1));
    std::tie(rms, xform) = qcp_rmsd_raw_vec<F>(iprod, E0, len, getrot);
    rms = rms.reshape({M, N});
    if (getrot) {
      // xform.slice(0,NULL).slice(0, 3)[3] = at::matmul(xform.slice(0, 3).slice(0, 3), -c1) + c2;
      xform = xform.reshape({M, N, 4, 4});
    }
  }
  if (getrot)
    return py::make_tuple(rms, xform);
  else
    return py::cast(rms);
}

// template <typename F, typename I>
// Tensor qcp_rmsd_regions(Tensor xyz1_in, Tensor xyz2_in, Tensor sizes, Tensor offsets, int junct = 0) {
//   Tensor rms({offsets.size(0)}, at::dtype<F>());
//   {
//     // py::gil_scoped_release release;
//     if (sizes.() != 1 || sizes.size(0) != offsets.size(0) || sizes.sum() != xyz2.rows())
//       throw std::runtime_error("bad sizes or offsets");
//     if (junct < 0) throw std::runtime_error("junct must be >= 0");
//
//     int nseg = sizes.cols();
//
//     Matrix<I, 1, Dynamic> offsets2(nseg);
//     offsets2.fill(0);
//     for (int i = 0; i < nseg - 1; ++i)
//       offsets2(0, i + 1) = offsets2(0, i) + sizes(0, i);
//
//     int ncrd = 0;
//     Matrix<F, 1, 3> cen2(0, 0, 0);
//     for (int iseg = 0; iseg < nseg; ++iseg) {
//       auto s = sizes(0, iseg);
//       ncrd += ((s > (2 * junct)) && junct > 0) ? 2 * junct : s;
//       add_to_center(cen2, xyz2, offsets2(0, iseg), sizes(0, iseg), junct);
//     }
//     cen2 /= ncrd; // xyz2.rows();
//     xyz2.rowwise() -= cen2;
//     F sqnorm2 = 0; // xyz2.array().square().sum();
//     for (int iseg = 0; iseg < nseg; ++iseg) {
//       add_to_sqnorm(sqnorm2, xyz2, offsets2(0, iseg), sizes(0, iseg), junct);
//     }
//
//     for (int ioff = 0; ioff < offsets.rows(); ++ioff) {
//       Matrix<F, 1, 3> cen1(0, 0, 0);
//       for (int iseg = 0; iseg < nseg; ++iseg) {
//         add_to_center(cen1, xyz1, offsets(ioff, iseg), sizes(0, iseg), junct);
//       }
//       cen1 /= ncrd; // xyz2.rows();
//       // if (junct > 0) {
//       // cout << "cen2 " << cen2 << endl;
//       // cout << "cen1 " << cen1 << endl;
//       // }
//
//       F sqnorm = 0;
//       Matrix<F, 3, 3> iprod;
//       iprod << 0, 0, 0, 0, 0, 0, 0, 0, 0;
//       for (int iseg = 0; iseg < nseg; ++iseg) {
//         add_to_iprod(iprod, sqnorm, xyz1, xyz2, offsets(ioff, iseg), offsets2(0, iseg), sizes(0, iseg),
//         cen1,
//                      junct);
//       }
//
//       double E0 = (sqnorm + sqnorm2) / 2;
//       double rmsd;
//       double A[9];
//       for (int ii = 0; ii < 3; ++ii)
//         for (int jj = 0; jj < 3; ++jj)
//           A[3 * ii + jj] = iprod(ii, jj);
//       qcp_calc_rmsd_maybe_rotation(NULL, A, &rmsd, E0, ncrd, -1);
//
//       rms[ioff] = rmsd;
//     }
//   }
//   py::capsule free_when_done(rms, [](void *f) { delete[] reinterpret_cast<F *>(f); });
//   return py::array_t<F>({offsets.rows()}, {sizeof(F)}, rms, free_when_done);
// }

template <typename F>
int qcp_calc_rmsd_maybe_rotation(F *rot, F *A, F *rmsd, F E0, F len, F minScore, int rotsize) {
  // cout << "INP " << A[0] << " " << E0 << " " << len << " " << minScore <<
  // endl;
  F Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz;
  F Szz2, Syy2, Sxx2, Sxy2, Syz2, Sxz2, Syx2, Szy2, Szx2, SyzSzymSyySzz2, Sxx2Syy2Szz2Syz2Szy2,
      Sxy2Sxz2Syx2Szx2, SxzpSzx, SyzpSzy, SxypSyx, SyzmSzy, SxzmSzx, SxymSyx, SxxpSyy, SxxmSyy;
  F C[4];
  int i;
  F mxEigenV;
  F oldg = 0.0;
  F b, a, delta, rms, qsqr;
  F q1, q2, q3, q4, normq;
  F a11, a12, a13, a14, a21, a22, a23, a24;
  F a31, a32, a33, a34, a41, a42, a43, a44;
  F a2, x2, y2, z2;
  F xy, az, zx, ay, yz, ax;
  F a3344_4334, a3244_4234, a3243_4233, a3143_4133, a3144_4134, a3142_4132;
  F evecprec, evalprec;
  if constexpr (std::is_same<F, float>::value) {
    evecprec = 1e-3;
    evalprec = 1e-6;
  } else if (std::is_same<F, double>::value) {
    evecprec = 1e-6;
    evalprec = 1e-11;
  } else {
    throw std::runtime_error("Only float and double supported");
  }

  Sxx = A[0];
  Sxy = A[1];
  Sxz = A[2];
  Syx = A[3];
  Syy = A[4];
  Syz = A[5];
  Szx = A[6];
  Szy = A[7];
  Szz = A[8];

  Sxx2 = Sxx * Sxx;
  Syy2 = Syy * Syy;
  Szz2 = Szz * Szz;

  Sxy2 = Sxy * Sxy;
  Syz2 = Syz * Syz;
  Sxz2 = Sxz * Sxz;

  Syx2 = Syx * Syx;
  Szy2 = Szy * Szy;
  Szx2 = Szx * Szx;

  SyzSzymSyySzz2 = 2.0 * (Syz * Szy - Syy * Szz);
  Sxx2Syy2Szz2Syz2Szy2 = Syy2 + Szz2 - Sxx2 + Syz2 + Szy2;

  C[2] = -2.0 * (Sxx2 + Syy2 + Szz2 + Sxy2 + Syx2 + Sxz2 + Szx2 + Syz2 + Szy2);
  C[1] = 8.0 * (Sxx * Syz * Szy + Syy * Szx * Sxz + Szz * Sxy * Syx - Sxx * Syy * Szz - Syz * Szx * Sxy -
                Szy * Syx * Sxz);

  SxzpSzx = Sxz + Szx;
  SyzpSzy = Syz + Szy;
  SxypSyx = Sxy + Syx;
  SyzmSzy = Syz - Szy;
  SxzmSzx = Sxz - Szx;
  SxymSyx = Sxy - Syx;
  SxxpSyy = Sxx + Syy;
  SxxmSyy = Sxx - Syy;
  Sxy2Sxz2Syx2Szx2 = Sxy2 + Sxz2 - Syx2 - Szx2;

  // cerr << SxzpSzx << " " << SyzpSzy << " " << SxypSyx << " " << SyzmSzy << " " << SxzmSzx << " " << SxymSyx
  // << " " << SxxpSyy << " " << SxxmSyy << endl;

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
  // cerr << C[0] << " " << C[1] << " " << C[2] << " " << E0 << endl;
  // Newton-Raphson
  mxEigenV = E0;
  for (i = 0; i < 50; ++i) {
    oldg = mxEigenV;
    x2 = mxEigenV * mxEigenV;
    b = (x2 + C[2]) * mxEigenV;
    a = b + C[1];
    delta = ((a * mxEigenV + C[0]) / (2.0 * x2 * mxEigenV + b + a));
    mxEigenV -= delta;
    // printf("\n diff[%3d]: %16g %16g %16g", i, mxEigenV - oldg,
    // evalprec*mxEigenV, mxEigenV);
    if (fabs(mxEigenV - oldg) < fabs(evalprec * mxEigenV)) break;
  }

  // cerr << mxEigenV << endl;

  // if (i == 50)
  // fprintf(stderr, "\nMore than %d iterations needed!\n", i);

  // the fabs() is to guard against extremely small, but *negative* numbers due
  // to floating point error
  rms = sqrt(fabs(2.0 * (E0 - mxEigenV) / len));
  (*rmsd) = rms;

  // cerr << rms << " " << len << endl;
  // printf("\n\n %16g %16g %16g \n", rms, E0, 2.0 * (E0 - mxEigenV)/len);

  if (minScore > 0 || rot == nullptr)
    if (rms < minScore || rot == nullptr) return (-1); // Don't bother with rotation.

  a11 = SxxpSyy + Szz - mxEigenV;
  a12 = SyzmSzy;
  a13 = -SxzmSzx;
  a14 = SxymSyx;
  a21 = SyzmSzy;
  a22 = SxxmSyy - Szz - mxEigenV;
  a23 = SxypSyx;
  a24 = SxzpSzx;
  a31 = a13;
  a32 = a23;
  a33 = Syy - Sxx - Szz - mxEigenV;
  a34 = SyzpSzy;
  a41 = a14;
  a42 = a24;
  a43 = a34;
  a44 = Szz - SxxpSyy - mxEigenV;
  a3344_4334 = a33 * a44 - a43 * a34;
  a3244_4234 = a32 * a44 - a42 * a34;
  a3243_4233 = a32 * a43 - a42 * a33;
  a3143_4133 = a31 * a43 - a41 * a33;
  a3144_4134 = a31 * a44 - a41 * a34;
  a3142_4132 = a31 * a42 - a41 * a32;
  q1 = a22 * a3344_4334 - a23 * a3244_4234 + a24 * a3243_4233;
  q2 = -a21 * a3344_4334 + a23 * a3144_4134 - a24 * a3143_4133;
  q3 = a21 * a3244_4234 - a22 * a3144_4134 + a24 * a3142_4132;
  q4 = -a21 * a3243_4233 + a22 * a3143_4133 - a23 * a3142_4132;

  qsqr = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4;

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
          rot[0] = rot[4] = rot[8] = 1.0;
          rot[1] = rot[2] = rot[3] = rot[5] = rot[6] = rot[7] = 0.0;

          return (0);
        }
      }
    }
  }

  normq = sqrt(qsqr);
  q1 /= normq;
  q2 /= normq;
  q3 /= normq;
  q4 /= normq;

  a2 = q1 * q1;
  x2 = q2 * q2;
  y2 = q3 * q3;
  z2 = q4 * q4;

  xy = q2 * q3;
  az = q1 * q4;
  zx = q4 * q2;
  ay = q1 * q3;
  yz = q3 * q4;
  ax = q1 * q2;

  if (rotsize == 4) {
    rot[0] = a2 + x2 - y2 - z2;
    rot[4] = 2 * (xy + az);
    rot[8] = 2 * (zx - ay);
    rot[1] = 2 * (xy - az);
    rot[5] = a2 - x2 + y2 - z2;
    rot[9] = 2 * (yz + ax);
    rot[2] = 2 * (zx + ay);
    rot[6] = 2 * (yz - ax);
    rot[10] = a2 - x2 - y2 + z2;
  } else {
    rot[0] = a2 + x2 - y2 - z2;
    rot[3] = 2 * (xy + az);
    rot[6] = 2 * (zx - ay);
    rot[1] = 2 * (xy - az);
    rot[4] = a2 - x2 + y2 - z2;
    rot[7] = 2 * (yz + ax);
    rot[2] = 2 * (zx + ay);
    rot[5] = 2 * (yz - ax);
    rot[8] = a2 - x2 - y2 + z2;
  }
  return (1);
}
PYBIND11_MODULE(_rms, m) {
  m.def("qcp_scan_cuda", &qcp_scan_cuda, "xyz1"_a, "xyz2"_a, "lbub"_a, "cyclic"_a = 1, "chainbreak"_a = 0,
        "lasu"_a = 0, "nthread"_a = 64, "rmsout"_a = false);

  m.def("qcp_rmsd_raw_vec_f4", &qcp_rmsd_raw_vec<float>, "iprod"_a, "E0"_a, "len"_a, "getrot"_a = false);
  m.def("qcp_rmsd_raw_vec_f8", &qcp_rmsd_raw_vec<double>, "iprod"_a, "E0"_a, "len"_a, "getrot"_a = false);

  m.def("qcp_rmsd_cuda", &qcp_calc_rmsd_cuda, "iprod"_a, "E0"_a, "len"_a, "getrot"_a = false,
        "nthread"_a = 128);
  m.def("qcp_rmsd_cuda_fixlen", &qcp_calc_rmsd_cuda_fixlen, "iprod"_a, "E0"_a, "len"_a, "getrot"_a = false,
        "nthread"_a = 128);

  m.def("qcp_rms_f4", &qcp_rmsd<float>, "xyz1"_a, "xyz2"_a);
  m.def("qcp_rms_f8", &qcp_rmsd<double>, "xyz1"_a, "xyz2"_a);

  m.def("qcp_rms_vec_f4", &qcp_rmsd_vec<float>, "xyz1"_a, "xyz2"_a, "getrot"_a = false);
  m.def("qcp_rms_vec_f8", &qcp_rmsd_vec<double>, "xyz1"_a, "xyz2"_a, "getrot"_a = false);

  // m.def("qcp_rms_align_vec_f4", &qcp_rmsd_align_vec<float>, "xyz1"_a,
  // "xyz2"_a);
  m.def("qcp_rms_align_f4", &qcp_rmsd_align<float>, "xyz1"_a, "xyz2"_a);
  m.def("qcp_rms_align_f8", &qcp_rmsd_align<double>, "xyz1"_a, "xyz2"_a);

  // m.def("qcp_rms_regions_f4i4", &qcp_rmsd_regions<float, int32_t>,
  // "xyz1"_a, "xyz2"_a, "sizes"_a, "offsets"_a, "junct"_a = 0);
  // m.def("qcp_rms_align_vec_f8", &qcp_rmsd_align_vec<double>);
}

} // namespace qcprms
} // namespace ipd
