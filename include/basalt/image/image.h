/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt-headers.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@file
@brief Image datatype with views, interpolation, gradients
*/

// This file is adapted from Pangolin. Original license:

/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) 2011 Steven Lovegrove
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <memory>

#include <Eigen/Dense>

#include <basalt/utils/assert.h>

// Renamed Pangoling defines to avoid clash
#define BASALT_HOST_DEVICE
#define BASALT_EXTENSION_IMAGE
#ifdef BASALT_ENABLE_BOUNDS_CHECKS
#define BASALT_BOUNDS_ASSERT(...) BASALT_ASSERT(__VA_ARGS__)
#else
#define BASALT_BOUNDS_ASSERT(...) ((void)0)
#endif

namespace basalt {

using byte = uint8_t;

template <typename T>
constexpr bool valid_image_type =
    std::is_same_v<T, uint16_t> || std::is_same_v<T, uint8_t> ||
    std::is_same_v<T, int32_t>;

/// @brief Helper class for copying objects.
template <typename T>
struct CopyObject {
  CopyObject(const T& obj) : obj(obj) {}
  const T& obj;
};

inline void PitchedCopy(void* dst_void, unsigned int dst_pitch_bytes,
                        const void* src_void, unsigned int src_pitch_bytes,
                        unsigned int width_bytes, unsigned int height) {
  byte* dst = (byte*)dst_void;
  byte* src = (byte*)src_void;
  if (dst_pitch_bytes == width_bytes && src_pitch_bytes == width_bytes) {
    std::memcpy(dst, src, height * width_bytes);
  } else {
    for (unsigned int row = 0; row < height; ++row) {
      std::memcpy(dst, src, width_bytes);
      dst += dst_pitch_bytes;
      src += src_pitch_bytes;
    }
  }
}

// evaluation is inspired by ceres-solver
// spline f is interpolated from 4 values.
// p0,1,2,3 are f(-1), f(0), f(1), f(2)
// x is between (0,1) the location where we want to evaluate spline
// f is spline evaluated at x. dfdx is derivative at x.
//
// Source:
//   https://github.com/ceres-solver/ceres-solver/blob/77c0c4d09c33f59f708ca0479aa2f1eb31fb6301/include/ceres/cubic_interpolation.h#L65
inline void CubHermiteSpline(const double& p0, const double& p1,
                             const double& p2, const double& p3, const double x,
                             double* f, double* dfdx) {
  const double a = 0.5 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3);
  const double b = 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3);
  const double c = 0.5 * (-p0 + p2);
  const double d = p1;
  // Horner scheme for: f = a x^3 + b x^2+ c x+d
  if (f != NULL) {
    *f = d + x * (c + x * (b + x * a));
  }

  // dfdx = 3a x^2 + 2b x + c
  if (dfdx != NULL) {
    *dfdx = c + x * (2.0 * b + 3.0 * a * x);
  }
}

/// @brief Image class that supports sub-images, interpolation, element access.
struct Image {
  enum Type { NONE, U8, U16, S32 };

  using LargestType = int32_t;

  inline Image() : pitch(0), ptr(0), w(0), h(0), t(U8) {}

  inline Image(void* ptr, size_t w, size_t h, size_t pitch, Type t)
      : pitch(pitch), ptr(ptr), w(w), h(h), t(t) {}

  BASALT_HOST_DEVICE inline size_t SizeBytes() const { return pitch * h; }

  BASALT_HOST_DEVICE inline size_t Area() const { return w * h; }

  BASALT_HOST_DEVICE inline bool IsValid() const { return ptr != 0; }

  static size_t TypeSize(Type t) {
    switch (t) {
      case U8: return 8 / 8;
      case U16: return 16 / 8;
      case S32: return 32 / 8;
      default: BASALT_ASSERT(false); return 0;
    }
  }

  size_t BytesPerPixel() const { return TypeSize(t); }

  BASALT_HOST_DEVICE inline bool IsContiguous() const {
    return w * BytesPerPixel() == pitch;
  }

  //////////////////////////////////////////////////////
  // Image transforms
  //////////////////////////////////////////////////////

 private:
  /// Using two template params to force caller to put the type explicitly
  template <typename T, typename UnaryOperation>
  BASALT_HOST_DEVICE inline void Transform_(UnaryOperation unary_op) {
    BASALT_ASSERT(IsValid());

    for (size_t y = 0; y < h; ++y) {
      T* el = (T*)RowPtr(y);
      const T* el_end = el + w;
      for (; el != el_end; ++el) {
        *el = unary_op(*el);
      }
    }
  }

  template <typename T>
  BASALT_HOST_DEVICE inline void Fill_(const T& val) {
    Transform_<T>([&](const T&) { return val; });
  }

  template <typename T>
  BASALT_HOST_DEVICE inline void Replace_(const T& oldval, const T& newval) {
    Transform_<T>([&](const T& val) { return (val == oldval) ? newval : val; });
  }

 public:
  inline void Fill(const LargestType& val) {
    switch (t) {
      case U8: Fill_<uint8_t>(val); break;
      case U16: Fill_<uint16_t>(val); break;
      case S32: Fill_<int32_t>(val); break;
      default: BASALT_ASSERT(false);
    }
  }

  inline void Replace(const LargestType& oldv, const LargestType& newv) {
    switch (t) {
      case U8: Replace_<uint8_t>(oldv, newv); break;
      case U16: Replace_<uint16_t>(oldv, newv); break;
      case S32: Replace_<int32_t>(oldv, newv); break;
      default: BASALT_ASSERT(false);
    }
  }

  inline void Memset(byte v = 0) {
    BASALT_ASSERT(IsValid());
    if (IsContiguous()) {
      std::memset(ptr, v, pitch * h);
    } else {
      for (size_t y = 0; y < h; ++y) {
        std::memset(RowPtr(y), v, pitch);
      }
    }
  }

  inline void CopyFrom(const Image& img) {
    if (IsValid() && img.IsValid() && t == img.t) {
      BASALT_ASSERT(w >= img.w && h >= img.h);
      PitchedCopy(ptr, pitch, img.ptr, img.pitch,
                  std::min(img.w, w) * BytesPerPixel(), std::min(img.h, h));
    } else if (img.IsValid() != IsValid()) {
      BASALT_ASSERT(false && "Cannot copy from / to an unasigned image.");
    }
  }

  //////////////////////////////////////////////////////
  // Reductions
  //////////////////////////////////////////////////////

  template <typename T>
  std::pair<T, T> MinMax() const {
    BASALT_ASSERT(IsValid());

    std::pair<T, T> minmax(std::numeric_limits<T>::max(),
                           std::numeric_limits<T>::lowest());
    for (size_t r = 0; r < h; ++r) {
      const T* ptr = RowPtr(r);
      const T* end = ptr + w;
      while (ptr != end) {
        minmax.first = std::min(*ptr, minmax.first);
        minmax.second = std::max(*ptr, minmax.second);
        ++ptr;
      }
    }
    return minmax;
  }

  template <typename T, typename Tout = T>
  Tout Sum() const {
    return Accumulate((T)0,
                      [](const T& lhs, const T& rhs) { return lhs + rhs; });
  }

  template <typename T, typename Tout = T>
  Tout Mean() const {
    return Sum<Tout>() / Area();
  }

  //////////////////////////////////////////////////////
  // Direct Pixel Access
  //////////////////////////////////////////////////////

  BASALT_HOST_DEVICE inline void* RowPtr(size_t y) {
    return &((byte*)ptr)[y * pitch];
  }

  BASALT_HOST_DEVICE inline const void* RowPtr(size_t y) const {
    return &((byte*)ptr)[y * pitch];
  }

  template <typename T>
  BASALT_HOST_DEVICE inline T& at(size_t x, size_t y) {
    BASALT_BOUNDS_ASSERT(InBounds(x, y));
    return ((T*)RowPtr(y))[x];
  }

  template <typename T>
  BASALT_HOST_DEVICE inline const T& at(size_t x, size_t y) const {
    BASALT_BOUNDS_ASSERT(InBounds(x, y));
    return ((T*)RowPtr(y))[x];
  }

  template <typename T, typename TVec>
  BASALT_HOST_DEVICE inline T& at(const TVec& p) {
    return at<T>(p[0], p[1]);
  }

  template <typename T, typename TVec>
  BASALT_HOST_DEVICE inline const T& at(const TVec& p) const {
    return at<T>(p[0], p[1]);
  }

  template <typename T>
  BASALT_HOST_DEVICE inline T& at(size_t ix) {
    T* elem = IsContiguous() ? &((T*)ptr)[ix] : &at<T>(ix % w, ix / w);
    BASALT_BOUNDS_ASSERT(InImage(elem));
    return *elem;
  }

  template <typename T>
  BASALT_HOST_DEVICE inline const T& at(size_t ix) const {
    T* elem = IsContiguous() ? &((T*)ptr)[ix] : &at<T>(ix % w, ix / w);
    BASALT_BOUNDS_ASSERT(InImage(elem));
    return *elem;
  }

  //////////////////////////////////////////////////////
  // Interpolated Pixel Access
  //////////////////////////////////////////////////////
  //
  // General notes on image interpolation and gradient computation:
  //
  // The default choice is bilinear image interpolation and bilinear
  // interpolation of the gradient image from central differences.
  //
  // We also have the exact gradient function of the bilinearly interpolated
  // image, which is less smooth and often less useful.
  //
  // We also have bicubic spline interpolation with exact gradients, which is
  // smoother and also has a smooth and exact gradient, but its more
  // computationally expensive. This interpolator is the same as used in
  // ceres-solver.
  //
  // For comparison of these approaches in case of Photometric BA, please refer
  // to Simon Klenk's Master's thesis (Section 4.3 Figures 4.9, 4.10):
  // https://vision.in.tum.de/_media/members/demmeln/klenk2020ma.pdf
  //
  //////////////////////////////////////////////////////

  // image value from bilinear interpolation (accessing 4 pixels)
  //
  // There is no bounds check (unless BASALT_ENABLE_BOUNDS_CHECKS is defined).
  // We assume that the pixel coordinates satisfy InBounds(x, y, 0) (the check
  // must be satisfied for floating point coordinates; in particular, this means
  // that integer arguments of w-1 for x or h-1 for y are out-of-bounds, see
  // notes on InBounds below). This also means there is no clamping of pixel
  // values.
  template <typename S>
  inline S interp(const Eigen::Matrix<S, 2, 1>& p) const {
    return interp<S>(p[0], p[1]);
  }

  // image value bilinearly. gradient from bilinear central differences
  // i.e. a central differences image is computed and gradient bilinearly
  // interpolated there (accessing 12 pixels)
  //
  // Note that the computed gradients are not exactly the gradients of the
  // interpolated function (unlike interpGradBilinearExact and
  // interpCubicSplines), but they are smoother than interpGradBilinearExact and
  // more efficient to compute than interpCubicSplines. In practice, these image
  // gradients are often a good choice for numerical optimization. However, they
  // cannot be used for numerical Jacobian unit tests.
  //
  // There is no bounds check (unless BASALT_ENABLE_BOUNDS_CHECKS is defined).
  // We assume that the pixel coordinates satisfy InBounds(x, y, 1). This also
  // means there is no clamping of pixel values.
  template <typename S>
  inline Eigen::Matrix<S, 3, 1> interpGrad(
      const Eigen::Matrix<S, 2, 1>& p) const {
    return interpGrad<S>(p[0], p[1]);
  }

  // image value and gradient from bilinear interpolation (accessing 4 pixels)
  //
  // This is the exact image gradient gradient function of the bilinearly
  // interpolated image. It is less smooth than alternatives and thus often not
  // as useful in practice.
  //
  // There is no bounds check (unless BASALT_ENABLE_BOUNDS_CHECKS is defined).
  // We assume that the pixel coordinates satisfy InBounds(x, y, 0). This also
  // means there is no clamping of pixel values.
  template <typename S>
  inline Eigen::Matrix<S, 3, 1> interpGradBilinearExact(
      const Eigen::Matrix<S, 2, 1>& p) const {
    return interpGradBilinearExact<S>(p[0], p[1]);
  }

  // image value from spline interpolation (accessing 16 pixels)
  // same as ceres-solver interpolation (see CubHermiteSpline above for source)
  //
  // This is the smoothest image interpolation alternative, but also the most
  // computationally expensive.
  //
  // There is no bounds check (unless BASALT_ENABLE_BOUNDS_CHECKS is defined).
  // We assume that the pixel coordinates satisfy InBounds(x, y, 0). Other pixel
  // values outside the image boundary that are needed for interpolation are
  // "clamped" to the boundary pixel values (following Ceres' implementation).
  template <typename S>
  inline double interpCubicSplines(const Eigen::Matrix<S, 2, 1>& p) const {
    return interpCubicSplines<S>(p[0], p[1]);
  }

  // image value and gradient from spline interpolation (accessing 16 pixels)
  // same as ceres-solver interpolation(see CubHermiteSpline above for source)
  //
  // This is the smoothest image interpolation and gradient alternative, but
  // also the most computationally expensive.
  //
  // There is no bounds check (unless BASALT_ENABLE_BOUNDS_CHECKS is defined).
  // We assume that the pixel coordinates satisfy InBounds(x, y, 0). Other pixel
  // values outside the image boundary that are needed for interpolation are
  // "clamped" to the boundary pixel values (following Ceres' implementation).
  template <typename S>
  inline Eigen::Matrix<S, 3, 1> interpGradCubicSplines(
      const Eigen::Matrix<S, 2, 1>& p) const {
    return interpGradCubicSplines<S>(p[0], p[1]);
  }

  // clamping on image border to stay consistent with ceres-solver
  void clamp(int& ixm1, int& ixp1, int& ixp2, int& iym1, int& iyp1,
             int& iyp2) const {
    // corner cases negative
    if (iym1 < 0) iym1 = 0;
    if (ixm1 < 0) ixm1 = 0;

    // corner cases positive
    if (ixp1 > (int)w - 1) ixp1 = w - 1;
    if (ixp2 > (int)w - 1) ixp2 = w - 1;
    if (iyp1 > (int)h - 1) iyp1 = h - 1;
    if (iyp2 > (int)h - 1) iyp2 = h - 1;
  }

 private:
  // for documentation see the alternative overload above
  template <typename T, typename S>
  inline S interp_(S x, S y) const {
    static_assert(std::is_floating_point_v<S>,
                  "interpolation / gradient only makes sense "
                  "for floating point result type");

    BASALT_BOUNDS_ASSERT(InBounds(x, y, 0));

    int ix = x;
    int iy = y;

    S dx = x - ix;
    S dy = y - iy;

    S ddx = S(1.0) - dx;
    S ddy = S(1.0) - dy;

    return ddx * ddy * at<T>(ix, iy) + ddx * dy * at<T>(ix, iy + 1) +
           dx * ddy * at<T>(ix + 1, iy) + dx * dy * at<T>(ix + 1, iy + 1);
  }

  // for documentation see the alternative overload above
  template <typename T, typename S>
  inline Eigen::Matrix<S, 3, 1> interpGrad_(S x, S y) const {
    static_assert(std::is_floating_point_v<S>,
                  "interpolation / gradient only makes sense "
                  "for floating point result type");

    BASALT_BOUNDS_ASSERT(InBounds(x, y, 1));

    int ix = x;
    int iy = y;

    S dx = x - ix;
    S dy = y - iy;

    S ddx = S(1.0) - dx;
    S ddy = S(1.0) - dy;

    Eigen::Matrix<S, 3, 1> res;

    const T& px0y0 = at<T>(ix, iy);
    const T& px1y0 = at<T>(ix + 1, iy);
    const T& px0y1 = at<T>(ix, iy + 1);
    const T& px1y1 = at<T>(ix + 1, iy + 1);

    res[0] = ddx * ddy * px0y0 + ddx * dy * px0y1 + dx * ddy * px1y0 +
             dx * dy * px1y1;

    const T& pxm1y0 = at<T>(ix - 1, iy);
    const T& pxm1y1 = at<T>(ix - 1, iy + 1);

    S res_mx = ddx * ddy * pxm1y0 + ddx * dy * pxm1y1 + dx * ddy * px0y0 +
               dx * dy * px0y1;

    const T& px2y0 = at<T>(ix + 2, iy);
    const T& px2y1 = at<T>(ix + 2, iy + 1);

    S res_px = ddx * ddy * px1y0 + ddx * dy * px1y1 + dx * ddy * px2y0 +
               dx * dy * px2y1;

    res[1] = S(0.5) * (res_px - res_mx);

    const T& px0ym1 = at<T>(ix, iy - 1);
    const T& px1ym1 = at<T>(ix + 1, iy - 1);

    S res_my = ddx * ddy * px0ym1 + ddx * dy * px0y0 + dx * ddy * px1ym1 +
               dx * dy * px1y0;

    const T& px0y2 = at<T>(ix, iy + 2);
    const T& px1y2 = at<T>(ix + 1, iy + 2);

    S res_py = ddx * ddy * px0y1 + ddx * dy * px0y2 + dx * ddy * px1y1 +
               dx * dy * px1y2;

    res[2] = S(0.5) * (res_py - res_my);

    return res;
  }

  // for documentation see the alternative overload above
  template <typename T, typename S>
  inline double interpCubicSplines_(S x, S y) const {
    static_assert(std::is_floating_point_v<S>,
                  "interpolation / gradient only makes sense "
                  "for floating point result type");

    BASALT_BOUNDS_ASSERT(InBounds(x, y, 0));

    double image_value;
    // p0,1,2,3 are f(-1), f(0), f(1), f(2) at pixel position
    double p0, p1, p2, p3;
    // f0,...f3 store function values at subpixel position
    double f0, f1, f2, f3;

    // get integer ix (column) and integer iy (row)
    int ix = x;
    int iy = y;
    // get x-1, x+1, x+2
    int ixm1 = ix - 1;
    int ixp1 = ix + 1;
    int ixp2 = ix + 2;
    // get y-1, y+1, y+2
    int iym1 = iy - 1;
    int iyp1 = iy + 1;
    int iyp2 = iy + 2;
    // clamp pixels to make it ceres-consistent
    clamp(ixm1, ixp1, ixp2, iym1, iyp1, iyp2);

    // row 0
    p0 = at<T>(ixm1, iym1);
    p1 = at<T>(ix, iym1);
    p2 = at<T>(ixp1, iym1);
    p3 = at<T>(ixp2, iym1);
    CubHermiteSpline(p0, p1, p2, p3, x - ix, &f0, NULL);

    // row 1
    p0 = at<T>(ixm1, iy);
    p1 = at<T>(ix, iy);
    p2 = at<T>(ixp1, iy);
    p3 = at<T>(ixp2, iy);
    CubHermiteSpline(p0, p1, p2, p3, x - ix, &f1, NULL);

    // row 2
    p0 = at<T>(ixm1, iyp1);
    p1 = at<T>(ix, iyp1);
    p2 = at<T>(ixp1, iyp1);
    p3 = at<T>(ixp2, iyp1);
    CubHermiteSpline(p0, p1, p2, p3, x - ix, &f2, NULL);

    // row 3
    p0 = at<T>(ixm1, iyp2);
    p1 = at<T>(ix, iyp2);
    p2 = at<T>(ixp1, iyp2);
    p3 = at<T>(ixp2, iyp2);
    CubHermiteSpline(p0, p1, p2, p3, x - ix, &f3, NULL);

    // now, interpolate vertically
    CubHermiteSpline(f0, f1, f2, f3, y - iy, &image_value, NULL);
    return image_value;
  }

  // for documentation see the alternative overload above
  template <typename T, typename S>
  inline Eigen::Matrix<S, 3, 1> interpGradCubicSplines_(S x, S y) const {
    static_assert(std::is_floating_point_v<S>,
                  "interpolation / gradient only makes sense "
                  "for floating point result type");

    BASALT_BOUNDS_ASSERT(InBounds(x, y, 0));

    Eigen::Matrix<S, 3, 1> res;

    // get integer x (column) and integer y (row)
    int ix = x;
    int iy = y;
    // p0,1,2,3 are f(-1), f(0), f(1), f(2) at pixel position
    double p0, p1, p2, p3;
    // f0,...f3 to store function values at subpixel position
    double f0, f1, f2, f3;
    // dfdx for horizontal interpolation along each row at subpixel
    double df0dx, df1dx, df2dx, df3dx;

    // get x-1, x+1, x+2
    int ixm1 = ix - 1;
    int ixp1 = ix + 1;
    int ixp2 = ix + 2;
    // get y-1, y+1, y+2
    int iym1 = iy - 1;
    int iyp1 = iy + 1;
    int iyp2 = iy + 2;
    // to make it consistent with Ceres Interpolator
    // we clamp the pixel values beyond image border
    clamp(ixm1, ixp1, ixp2, iym1, iyp1, iyp2);

    // row 0
    p0 = at<T>(ixm1, iym1);
    p1 = at<T>(ix, iym1);
    p2 = at<T>(ixp1, iym1);
    p3 = at<T>(ixp2, iym1);
    CubHermiteSpline(p0, p1, p2, p3, x - ix, &f0, &df0dx);

    // row 1
    p0 = at<T>(ixm1, iy);
    p1 = at<T>(ix, iy);
    p2 = at<T>(ixp1, iy);
    p3 = at<T>(ixp2, iy);
    CubHermiteSpline(p0, p1, p2, p3, x - ix, &f1, &df1dx);

    // row 2
    p0 = at<T>(ixm1, iyp1);
    p1 = at<T>(ix, iyp1);
    p2 = at<T>(ixp1, iyp1);
    p3 = at<T>(ixp2, iyp1);
    CubHermiteSpline(p0, p1, p2, p3, x - ix, &f2, &df2dx);

    // row 3
    p0 = at<T>(ixm1, iyp2);
    p1 = at<T>(ix, iyp2);
    p2 = at<T>(ixp1, iyp2);
    p3 = at<T>(ixp2, iyp2);
    CubHermiteSpline(p0, p1, p2, p3, x - ix, &f3, &df3dx);

    // now, interpolate vertically
    CubHermiteSpline(f0, f1, f2, f3, y - iy, &res[0], &res[2]);
    CubHermiteSpline(df0dx, df1dx, df2dx, df3dx, y - iy, &res[1], NULL);

    return res;
  }

  // for documentation see the alternative overload above
  template <typename T, typename S>
  inline Eigen::Matrix<S, 3, 1> interpGradBilinearExact_(S x, S y) const {
    static_assert(std::is_floating_point_v<S>,
                  "interpolation / gradient only makes sense "
                  "for floating point result type");

    BASALT_BOUNDS_ASSERT(InBounds(x, y, 0));

    Eigen::Matrix<S, 3, 1> res;

    // getting integer coordinates
    int ix = x;
    int iy = y;
    S dx = x - ix;
    S dy = y - iy;
    S ddx = 1.0f - dx;
    S ddy = 1.0f - dy;
    // clamping pixel values
    int ixp1 = ix + 1;
    int iyp1 = iy + 1;
    if (ixp1 > (int)w - 1) ixp1 = w - 1;
    if (iyp1 > (int)h - 1) iyp1 = h - 1;

    // we only require 4 coordinates
    // for bilinear interpolation
    const T& px0y0 = at<T>(ix, iy);
    const T& px1y0 = at<T>(ixp1, iy);
    const T& px0y1 = at<T>(ix, iyp1);
    const T& px1y1 = at<T>(ixp1, iyp1);

    // interpolate in x direction (fix y)
    const double fxy0 = ddx * px0y0 + dx * px1y0;
    const double fxy1 = ddx * px0y1 + dx * px1y1;

    // image value: normal bilinear interpolation
    res[0] = fxy0 * ddy + fxy1 * dy;

    // derivative in y direction
    res[2] = fxy1 - fxy0;

    // interpolate in y direction (fix x)
    const double fx0y = ddy * px0y0 + dy * px0y1;
    const double fx1y = ddy * px1y0 + dy * px1y1;

    // derivative in x direction
    res[1] = fx1y - fx0y;

    return res;
  }

 public:
  //////////////////////////////////////////////////////
  // Dynamic template dispatch
  //////////////////////////////////////////////////////

  template <typename S>
  inline S interp(S x, S y) const {
    switch (t) {
      case U8: return interp_<uint8_t, S>(x, y);
      case U16: return interp_<uint16_t, S>(x, y);
      case S32: return interp_<int32_t, S>(x, y);
      default: BASALT_ASSERT(false); return {};
    }
  }

  template <typename S>
  inline Eigen::Matrix<S, 3, 1> interpGrad(S x, S y) const {
    switch (t) {
      case U8: return interpGrad_<uint8_t, S>(x, y);
      case U16: return interpGrad_<uint16_t, S>(x, y);
      case S32: return interpGrad_<int32_t, S>(x, y);
      default: BASALT_ASSERT(false); return {};
    }
  }

  template <typename S>
  inline Eigen::Matrix<S, 3, 1> interpGradBilinearExact(S x, S y) const {
    switch (t) {
      case U8: return interpGradBilinearExact_<uint8_t, S>(x, y);
      case U16: return interpGradBilinearExact_<uint16_t, S>(x, y);
      case S32: return interpGradBilinearExact_<int32_t, S>(x, y);
      default: BASALT_ASSERT(false); return {};
    }
  }

  template <typename S>
  inline double interpCubicSplines(S x, S y) const {
    switch (t) {
      case U8: return interpCubicSplines_<uint8_t, S>(x, y);
      case U16: return interpCubicSplines_<uint16_t, S>(x, y);
      case S32: return interpCubicSplines_<int32_t, S>(x, y);
      default: BASALT_ASSERT(false); return {};
    }
  }

  template <typename S>
  inline Eigen::Matrix<S, 3, 1> interpGradCubicSplines(S x, S y) const {
    switch (t) {
      case U8: return interpGradCubicSplines_<uint8_t, S>(x, y);
      case U16: return interpGradCubicSplines_<uint16_t, S>(x, y);
      case S32: return interpGradCubicSplines_<int32_t, S>(x, y);
      default: BASALT_ASSERT(false); return {};
    }
  }

  //////////////////////////////////////////////////////
  // Bounds Checking
  //////////////////////////////////////////////////////
  // General note on bounds checking:
  //
  // For integer coordinates, all valid pixels in the image are in bounds, i.e.
  // values [0, ..., w-1] for x and [0, ..., h-1] for y.
  //
  // For floating point coordinates, we have a slightly different definition for
  // "in bounds". Coordinates are in bounds, if they have 4 neighbording integer
  // pixels that are in bounds, i.e. if they can be bilinearly interpolated.
  // Neighboring integer coordinates for a floating point coordinate fx are
  // defined as ((int)fx, (intfx)+1). In particular, this means that floating
  // point x coordinates >= w-1 are out of bounds, and similarly y coordinates
  // >= h-1 (notice ">=", not ">").
  //
  // With this definition, calling interp(x, y) is valid iff InBounds(x, y, 0)
  // is true, und interpGrad(x, y) is valid iff InBounds(x, y, 1) is true.
  //
  // Pixel access, interpolation and gradient computation don't have bounds
  // checkes by default for performance reasons, but for debugging, bounds
  // checks can be enabled by defining BASALT_ENABLE_BOUNDS_CHECKS.
  //////////////////////////////////////////////////////

  /// Test if pointer lies within dense region of image memory.
  BASALT_HOST_DEVICE
  bool InImage(const void* ptest) const {
    size_t idx_in_bytes = (byte*)ptest - (byte*)ptr;
    bool in_valid_pitch = idx_in_bytes % pitch < w * BytesPerPixel();
    bool within_mem_bounds = ptr <= ptest && ptest < RowPtr(h);
    return within_mem_bounds && in_valid_pitch;
  }

  /// In bounds check for integer coordinates.
  BASALT_HOST_DEVICE inline bool InBounds(int x, int y) const {
    return 0 <= x && x < (int)w && 0 <= y && y < (int)h;
  }

  /// In bounds check for floating point coordinates with given border.
  /// See note above for exact definition of "in bounds".
  BASALT_HOST_DEVICE inline bool InBounds(float x, float y,
                                          float border) const {
    return border <= x && x < (w - border - 1) && border <= y &&
           y < (h - border - 1);
  }

  /// In bounds check for integer or floating point coordinates with given
  /// border. See note above for exact definition of "in bounds", which is
  /// different for integers and floats.
  template <typename Derived>
  BASALT_HOST_DEVICE inline bool InBounds(
      const Eigen::MatrixBase<Derived>& p,
      const typename Derived::Scalar border) const {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 2);

    using Scalar = typename Derived::Scalar;

    Scalar offset(0);
    if constexpr (std::is_floating_point_v<Scalar>) {
      offset = Scalar(1);
    }

    return border <= p[0] && p[0] < ((int)w - border - offset) &&
           border <= p[1] && p[1] < ((int)h - border - offset);
  }

  //////////////////////////////////////////////////////
  // Obtain slices / subimages
  //////////////////////////////////////////////////////

  BASALT_HOST_DEVICE inline const Image SubImage(size_t x, size_t y,
                                                 size_t width,
                                                 size_t height) const {
    BASALT_ASSERT((x + width) <= w && (y + height) <= h);
    void* base = &((byte*)RowPtr(y))[x * BytesPerPixel()];
    return Image(base, width, height, pitch, t);
  }

  BASALT_HOST_DEVICE inline Image SubImage(size_t x, size_t y, size_t width,
                                           size_t height) {
    BASALT_ASSERT((x + width) <= w && (y + height) <= h);
    void* base = &((byte*)RowPtr(y))[x * BytesPerPixel()];
    return Image(base, width, height, pitch, t);
  }

  BASALT_HOST_DEVICE inline Image Row(int y) const {
    return SubImage(0, y, w, 1);
  }

  BASALT_HOST_DEVICE inline Image Col(int x) const {
    return SubImage(x, 0, 1, h);
  }

  //////////////////////////////////////////////////////
  // Data members
  //////////////////////////////////////////////////////

  size_t pitch;
  void* ptr;
  size_t w;
  size_t h;
  Type t;

  BASALT_EXTENSION_IMAGE
};

/// @brief Image that manages it's own memory, storing a strong pointer to it's
/// memory. Note that this class does not do reference counting. Wrap it with
/// a shared_ptr for that @see ManagedImage::Ptr.
class ManagedImage : public Image {
 public:
  using Ptr = std::shared_ptr<ManagedImage>;

  struct Allocator {
    Type t;
    Allocator(Type t) : t(t) {}

    void* allocate(size_t n) {
      switch (t) {
        case U8: return std::allocator<uint8_t>().allocate(n);
        case U16: return std::allocator<uint16_t>().allocate(n);
        case S32: return std::allocator<int32_t>().allocate(n);
        default: BASALT_ASSERT(false);
      }
    }

    void deallocate(void* ptr, size_t n) {
      switch (t) {
        case U8: std::allocator<uint8_t>().deallocate((uint8_t*)ptr, n); break;
        case U16:
          std::allocator<uint16_t>().deallocate((uint16_t*)ptr, n);
          break;
        case S32: std::allocator<int32_t>().deallocate((int32_t*)ptr, n); break;
        default: BASALT_ASSERT(false);
      }
    }
  };

  // Destructor
  inline ~ManagedImage() { Deallocate(); }

  // Null image
  inline ManagedImage() {}

  // Row image
  inline ManagedImage(size_t w, Type t)
      : Image(Allocator(t).allocate(w), w, 1, w * TypeSize(t), t) {}

  inline ManagedImage(size_t w, size_t h, Type t)
      : Image(Allocator(t).allocate(w * h), w, h, w * TypeSize(t), t) {}

  inline ManagedImage(size_t w, size_t h, size_t pitch_bytes, Type t)
      : Image(Allocator(t).allocate((h * pitch_bytes) / TypeSize(t) + 1), w, h,
              pitch_bytes, t) {}

  /// Constructor with data pointer. @p owner is expected to be an object that,
  /// when released, frees the @p ptr memory. If null, we'll free it.
  inline ManagedImage(size_t w, size_t h, size_t pitch, Type t, void* ptr,
                      std::shared_ptr<void> owner = nullptr)
      : Image(ptr, w, h, pitch, t), owner(owner) {}

  // Not copy constructable
  inline ManagedImage(const ManagedImage& other) = delete;

  // Move constructor
  inline ManagedImage(ManagedImage&& img) { *this = std::move(img); }

  // Move asignment
  inline void operator=(ManagedImage&& img) {
    Deallocate();
    Image::pitch = img.pitch;
    Image::ptr = img.ptr;
    Image::w = img.w;
    Image::h = img.h;
    Image::t = img.t;
    owner = img.owner;
    img.ptr = nullptr;
    img.owner = nullptr;
  }

  // Explicit copy constructor
  template <typename TOther>
  ManagedImage(const CopyObject<TOther>& other) {
    CopyFrom(other.obj);
  }

  // Explicit copy assignment
  template <typename TOther>
  void operator=(const CopyObject<TOther>& other) {
    CopyFrom(other.obj);
  }

  inline void Swap(ManagedImage& img) {
    std::swap(img.pitch, Image::pitch);
    std::swap(img.ptr, Image::ptr);
    std::swap(img.w, Image::w);
    std::swap(img.h, Image::h);
    std::swap(img.t, Image::t);
    std::swap(img.owner, owner);
  }

  inline void CopyFrom(const Image& img) {
    if (!Image::IsValid() || w != img.w || h != img.h || t != img.t) {
      Reinitialise(img.w, img.h, img.t);
    }
    Image::CopyFrom(img);
  }

  inline void Reinitialise(size_t w, size_t h, Type t) {
    if (!Image::ptr || Image::w != w || Image::h != h || Image::t != t) {
      *this = ManagedImage(w, h, t);
    }
  }

  inline void Reinitialise(size_t w, size_t h, size_t pitch, Type t) {
    if (!Image::ptr || Image::w != w || Image::h != h ||
        Image::pitch != pitch || Image::t != t) {
      *this = ManagedImage(w, h, pitch, t);
    }
  }

  inline void Deallocate() {
    if (ptr && owner == nullptr) {
      Allocator(t).deallocate(ptr, (h * pitch) / BytesPerPixel());
      ptr = nullptr;
    }
    owner = nullptr;
  }

  inline void operator-=(const ManagedImage& img) {
    BASALT_ASSERT(ptr && img.ptr && w == img.w && h == img.h && t == img.t);
    switch (t) {
      case U8: operator-=<uint8_t>(img); break;
      case U16: operator-=<uint16_t>(img); break;
      case S32: operator-=<int32_t>(img); break;
      default: BASALT_ASSERT(false);
    }
  }

 private:
  template <typename T>
  inline void operator-=(const ManagedImage& img) {
    for (size_t j = 0; j < img.h; j++) {
      T* this_row = (T*)this->RowPtr(j);
      const T* other_row = (T*)img.RowPtr(j);
      for (size_t i = 0; i < img.w; i++) {
        this_row[i] -= other_row[i];
      }
    }
  }

  /// If null, we deallocate @ref ptr memory ourselves. Otherwise it's a pointer
  /// to a owner object that will free @ref ptr on its destructor.
  std::shared_ptr<void> owner = nullptr;
};

}  // namespace basalt
