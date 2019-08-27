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
*/

#pragma once

#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>

namespace cereal {

// NOTE: Serialization functions for non-basalt types are marked static to
// ensure internal linkage, which allows different and incompatible definitions
// in other libraries that also include basalt-headers (as long as they are not
// included in the same translation unit). See
// https://groups.google.com/d/msg/cerealcpp/WswQi_Sh-bw/Pw0GrfIqFQAJ for a more
// detailed discussion.

template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options,
          int _MaxRows, int _MaxCols>
static inline
    typename std::enable_if<_Rows != Eigen::Dynamic && _Cols != Eigen::Dynamic,
                            void>::type
    serialize(
        Archive& archive,
        Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& m) {
  static_assert(_Rows > 0, "matrix should be static size");
  static_assert(_Cols > 0, "matrix should be static size");
  cereal::size_type s = _Rows * _Cols;
  archive(cereal::make_size_tag(s));
  if (s != _Rows * _Cols) {
    throw std::runtime_error("matrix has incorrect length");
  }
  for (size_t i = 0; i < _Rows; i++)
    for (size_t j = 0; j < _Cols; j++) archive(m(i, j));
}

template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options,
          int _MaxRows, int _MaxCols>
static inline
    typename std::enable_if<_Rows == Eigen::Dynamic || _Cols == Eigen::Dynamic,
                            void>::type
    save(Archive &ar, const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options,
                                          _MaxRows, _MaxCols> &matrix) {
  const std::int32_t rows = static_cast<std::int32_t>(matrix.rows());
  const std::int32_t cols = static_cast<std::int32_t>(matrix.cols());
  ar(rows);
  ar(cols);
  ar(binary_data(matrix.data(), rows * cols * sizeof(_Scalar)));
};

template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options,
          int _MaxRows, int _MaxCols>
static inline
    typename std::enable_if<_Rows == Eigen::Dynamic || _Cols == Eigen::Dynamic,
                            void>::type
    load(Archive &ar,
         Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>
             &matrix) {
  std::int32_t rows;
  std::int32_t cols;
  ar(rows);
  ar(cols);

  matrix.resize(rows, cols);

  ar(binary_data(matrix.data(),
                 static_cast<std::size_t>(rows * cols * sizeof(_Scalar))));
};

template <class Archive, class Scalar>
static inline void serialize(Archive& ar, Sophus::SE3<Scalar>& p) {
  ar(cereal::make_nvp("px", p.translation()[0]),
     cereal::make_nvp("py", p.translation()[1]),
     cereal::make_nvp("pz", p.translation()[2]),
     cereal::make_nvp("qx", p.so3().data()[0]),
     cereal::make_nvp("qy", p.so3().data()[1]),
     cereal::make_nvp("qz", p.so3().data()[2]),
     cereal::make_nvp("qw", p.so3().data()[3]));
}

}  // namespace cereal
