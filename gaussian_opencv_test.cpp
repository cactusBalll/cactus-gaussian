#include "gaussian_opencv.h"
#include <random>
#include <iostream>

constexpr int w = 4096;
constexpr int h = 4096;
int main() {
  std::random_device rd;
  std::mt19937 gen(rd());
  float *random_image = new float[w * h];
  float *out_buffer = new float[w * h];
  std::cout << "test begin." << std::endl;
  // 生成随机数据
  for (int i = 0; i < w * h; i++) {
    random_image[i] = std::generate_canonical<float, 10>(gen);
  }
  std::cout << "data generated." << std::endl;
  gaussian_cv2_simd(out_buffer, random_image, h, w, 1.6);
  gaussian_cv2_cuda(out_buffer, random_image, h, w, 1.6);
  delete []random_image;
  delete []out_buffer;
  return 0;
}