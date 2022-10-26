/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt-headers.git

Copyright (c) 2022, Collabora Ltd.
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
@brief Interoperability between OpenCV cv::Mat datatype and our own Image types
*/

#pragma once

#include <basalt/image/image.h>
#include <basalt/utils/assert.h>

#include <opencv4/opencv2/core.hpp>

#include <memory>

namespace basalt {

inline Image::Type cv_as_mimg_type(int cvtype) {
  switch (cvtype) {
    case CV_8UC1: return Image::U8;
    case CV_16UC1: return Image::U16;
    default: BASALT_ASSERT_MSG(false, "Not implemented");
  }
  return Image::NONE;
}

inline int mimg_as_cv_type(Image::Type mimg_type) {
  switch (mimg_type) {
    case Image::U8: return CV_8UC1;
    case Image::U16: return CV_16UC1;
    default: BASALT_ASSERT_MSG(false, "Not implemented");
  }
  return -1;
}

/// Copies cv::Mat contents into a new @ref ManagedImage.
inline ManagedImage::Ptr import_cvmat_copy(
    cv::Mat img, Image::Type mimg_type = Image::NONE) {
  if (mimg_type == Image::NONE) mimg_type = cv_as_mimg_type(img.type());
  auto mimg = std::make_shared<ManagedImage>(img.cols, img.rows, mimg_type);

  if (img.type() == CV_8UC1 && mimg_type == Image::U8)
    for (int y = 0; y < img.rows; y++)
      for (int x = 0; x < img.cols; x++)
        mimg->at<uint8_t>(x, y) = img.at<uint8_t>(y, x);
  else if (img.type() == CV_8UC1 && mimg_type == Image::U16)
    for (int y = 0; y < img.rows; y++)
      for (int x = 0; x < img.cols; x++)
        mimg->at<uint16_t>(x, y) = img.at<uint8_t>(y, x) << 8;
  else
    BASALT_ASSERT_MSG(false, "Not implemented");

  return mimg;
}

/// Creates a @ref ManagedImage from cv::Mat. Tries to avoid copying if possible
inline ManagedImage::Ptr import_cvmat(cv::Mat cvimg,
                                      Image::Type mimg_type = Image::NONE) {
  // If target type doesn't match, we will need to copy and convert:
  if (mimg_type != Image::NONE && mimg_type != cv_as_mimg_type(cvimg.type()))
    return import_cvmat_copy(cvimg, mimg_type);

  // Otherwise, we can just reference cvimg memory without a copy:

  // Helper class to keep track of the cv::Mat refcounting and destruction
  struct CvMatHolder {
    cv::Mat mat;
    CvMatHolder(cv::Mat mat) : mat(mat) {}
  };
  auto mh = std::make_shared<CvMatHolder>(cvimg);

  size_t w = cvimg.cols;
  size_t h = cvimg.rows;
  size_t pitch = cvimg.step;
  Image::Type t = cv_as_mimg_type(cvimg.type());
  auto mimg = std::make_shared<ManagedImage>(w, h, pitch, t, cvimg.ptr(), mh);
  BASALT_ASSERT(mimg->SizeBytes() == h * pitch);

  return mimg;
}

/// Creates a cv::Mat of type @p cvtype out of an @ref Image with a copy.
inline cv::Mat export_cvmat_copy(const Image& img, int cvtype = -1) {
  if (cvtype == -1) cvtype = mimg_as_cv_type(img.t);

  cv::Mat cvimg(img.h, img.w, cvtype);

  if (img.t == Image::U8 && cvtype == CV_8UC1)
    for (size_t y = 0; y < img.h; y++)
      for (size_t x = 0; x < img.w; x++)
        cvimg.at<uint8_t>(y, x) = img.at<uint8_t>(x, y);
  else if (img.t == Image::U16 && cvtype == CV_8UC1)
    for (size_t y = 0; y < img.h; y++)
      for (size_t x = 0; x < img.w; x++)
        cvimg.at<uint8_t>(y, x) = img.at<uint16_t>(x, y) >> 8;
  else
    BASALT_ASSERT_MSG(false, "Not implemented");

  return cvimg;
}

/// Creates a cv::Mat out of an @ref Image. Tries to avoid copying if possible.
inline cv::Mat export_cvmat(const Image& img, int cvtype = -1) {
  if (cvtype != -1 && cvtype != mimg_as_cv_type(img.t))
    return export_cvmat_copy(img, cvtype);

  return cv::Mat(img.h, img.w, mimg_as_cv_type(img.t), img.ptr, img.pitch);
}

}  // namespace basalt
