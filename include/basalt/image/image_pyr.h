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
@brief Image pyramid implementation stored as mipmap
*/

#pragma once

#include <vector>

#include <basalt/image/image.h>

namespace basalt {

/// @brief Image pyramid stored in separate and progressively smaller
/// ManagedImages
///
/// \image html mipmap.jpeg
/// Computes image pyramid (see \ref subsample) and stores the subsamples
/// (https://en.wikipedia.org/wiki/Mipmap).
class ManagedImagePyr {
 public:
  using Ptr = std::shared_ptr<ManagedImagePyr>;

  /// @brief Default constructor.
  inline ManagedImagePyr() {}

  /// @brief Construct image pyramid from other image.
  ///
  /// @param img image to use for the pyramid level 0
  /// @param num_level number of levels for the pyramid (excluding level 0)
  inline ManagedImagePyr(ManagedImage::Ptr img, size_t num_levels) {
    setFromImage(img, num_levels);
  }

  /// @brief Create image pyramid from @p img.
  ///
  /// @param img image to use for the pyramid level 0
  /// @param num_level number of levels for the pyramid
  inline void setFromImage(ManagedImage::Ptr img, size_t num_levels) {
    levels = std::vector<ManagedImage::Ptr>(num_levels + 1);
    levels[0] = img;

    for (size_t i = 1; i <= num_levels; i++) {
      size_t w = img->w >> i;
      size_t h = img->h >> i;
      levels[i] = std::make_shared<ManagedImage>(w, h, img->t);
    };

    for (size_t i = 0; i < num_levels; i++) {
      const Image& l = *levels[i];
      Image& lp1 = *levels[i + 1];
      subsample(l, lp1);
    }
  }

  /// @brief Extrapolate image after border with reflection.
  static inline int border101(int x, int h) {
    return h - 1 - std::abs(h - 1 - x);
  }

  static void subsample(const Image& img, Image& img_sub) {
    BASALT_ASSERT(img.t == img_sub.t);
    switch (img.t) {
      case Image::U8: subsample<uint8_t>(img, img_sub); break;
      case Image::U16: subsample<uint16_t>(img, img_sub); break;
      case Image::S32: subsample<int32_t>(img, img_sub); break;
      default: BASALT_ASSERT(false);
    }
  }

  /// @brief Subsample the image twice in each direction.
  ///
  /// Subsampling is done by convolution with Gaussian kernel
  /// \f[
  /// \frac{1}{256}
  /// \begin{bmatrix}
  ///   1 & 4 & 6 & 4 & 1
  /// \\4 & 16 & 24 & 16 & 4
  /// \\6 & 24 & 36 & 24 & 6
  /// \\4 & 16 & 24 & 16 & 4
  /// \\1 & 4 & 6 & 4 & 1
  /// \\ \end{bmatrix}
  /// \f]
  /// and removing every even-numbered row and column.
  template <typename T>
  static void subsample(const Image& img, Image& img_sub) {
    static_assert(valid_image_type<T>, "Unsupported type");

    constexpr int kernel[5] = {1, 4, 6, 4, 1};

    // accumulator
    ManagedImage tmp(img_sub.h, img.w, Image::S32);

    // Vertical convolution
    {
      for (int r = 0; r < int(img_sub.h); r++) {
        const T* row_m2 = (T*)img.RowPtr(std::abs(2 * r - 2));
        const T* row_m1 = (T*)img.RowPtr(std::abs(2 * r - 1));
        const T* row = (T*)img.RowPtr(2 * r);
        const T* row_p1 = (T*)img.RowPtr(border101(2 * r + 1, img.h));
        const T* row_p2 = (T*)img.RowPtr(border101(2 * r + 2, img.h));

        for (int c = 0; c < int(img.w); c++) {
          tmp.at<int32_t>(r, c) =
              kernel[0] * int(row_m2[c]) + kernel[1] * int(row_m1[c]) +
              kernel[2] * int(row[c]) + kernel[3] * int(row_p1[c]) +
              kernel[4] * int(row_p2[c]);
        }
      }
    }

    // Horizontal convolution
    {
      for (int c = 0; c < int(img_sub.w); c++) {
        const int* row_m2 = (int*)tmp.RowPtr(std::abs(2 * c - 2));
        const int* row_m1 = (int*)tmp.RowPtr(std::abs(2 * c - 1));
        const int* row = (int*)tmp.RowPtr(2 * c);
        const int* row_p1 = (int*)tmp.RowPtr(border101(2 * c + 1, tmp.h));
        const int* row_p2 = (int*)tmp.RowPtr(border101(2 * c + 2, tmp.h));

        for (int r = 0; r < int(tmp.w); r++) {
          int val_int = kernel[0] * row_m2[r] + kernel[1] * row_m1[r] +
                        kernel[2] * row[r] + kernel[3] * row_p1[r] +
                        kernel[4] * row_p2[r];
          T val = ((val_int + (1 << 7)) >> 8);
          img_sub.at<T>(c, r) = val;
        }
      }
    }
  }

  /// @brief Return const image of the certain level
  ///
  /// @param lvl level to return
  /// @return const image of with the pyramid level
  inline const Image& lvl(size_t lvl) const { return *levels[lvl]; }

 protected:
  std::vector<ManagedImage::Ptr> levels;
};

}  // namespace basalt
