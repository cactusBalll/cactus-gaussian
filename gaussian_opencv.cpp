#include "gaussian_opencv.h"
#include "common.h"
#include "opencv2/core/cuda/common.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace cv::cuda;

// 仅用于测试opencv+cuda，但是此种方式不被支持，只能重新编译opencv
// // 前向声明
// namespace filter {
// template <typename T, typename D>
// void linearRow(PtrStepSzb src, PtrStepSzb dst, const float *kernel, int
// ksize,
//                int anchor, int brd_type, int cc, cudaStream_t stream);

// template <typename T, typename D>
// void linearColumn(PtrStepSzb src, PtrStepSzb dst, const float *kernel,
//                   int ksize, int anchor, int brd_type, int cc,
//                   cudaStream_t stream);
// template void linearRow<float, float>(PtrStepSzb src, PtrStepSzb dst,
//                                       const float *kernel, int ksize,
//                                       int anchor, int brd_type, int cc,
//                                       cudaStream_t stream);
// template void linearColumn<float, float>(PtrStepSzb src, PtrStepSzb dst,
//                                          const float *kernel, int ksize,
//                                          int anchor, int brd_type, int cc,
//                                          cudaStream_t stream);
// } // namespace filter

void gaussian_cv2_simd(float *dest, float *src, int image_h, int image_w,
                       float sigma) {
  auto kernel_vec = make_gaussian_kernel_1d(sigma);
  Mat kernel = Mat(kernel_vec, true);
  Mat src_mat(image_h, image_w, CV_32F, src);
  Mat dest_mat{image_h, image_w, CV_32F, dest};
  sepFilter2D(src_mat, dest_mat, CV_32F, kernel.t(), kernel, Point(-1, -1), 0,
              BORDER_CONSTANT);
}

void gaussian_cv2_cuda(float *dest, float *src, int image_h, int image_w,
                       float sigma) {
  auto kernel_vec = make_gaussian_kernel_1d(sigma);
  Mat kernel = Mat(kernel_vec, true);
  GpuMat src_mat_d{};
  // GpuMat mid_mat_d(image_h, image_w, CV_32F);
  GpuMat dest_mat_d(image_h, image_w, CV_32F);
  // GpuMat kernel_mat_d{};
  Mat src_mat(image_h, image_w, CV_32F, src);
  Mat dest_mat{image_h, image_w, CV_32F, dest};
  //使用opencv API上传图像矩阵至设备
  src_mat_d.upload(src_mat);
  // kernel_mat_d.upload(kernel);
  // 89 - Compute Capability 8.9
  // 8  - 卷积核中间位置anchor
  // filter::linearRow<float, float>(src_mat_d, mid_mat_d,
  //                                 kernel_mat_d.ptr<float>(), KERNEL_SIZE, 8,
  //                                 BORDER_CONSTANT, 89, 0);
  // filter::linearRow<float, float>(mid_mat_d, dest_mat_d,
  //                                   kernel_mat_d.ptr<float>(), KERNEL_SIZE,
  //                                   8, BORDER_CONSTANT, 89, 0);
  // 使用封装版本
  auto flt = cuda::createSeparableLinearFilter(
      CV_32F, CV_32F, kernel, kernel, cv::Point(-1, -1), BORDER_CONSTANT,
      BORDER_CONSTANT);
  flt->apply(src_mat_d, dest_mat_d);
  // 把结果从device下载到host
  dest_mat_d.download(dest_mat);
}