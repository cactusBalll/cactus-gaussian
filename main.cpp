#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>

#include "gaussian.h"
#include "gaussian_opencv.h"

constexpr int w = 4096;
constexpr int h = 4096;
/// @brief  速度测试尝试求平均次数，设置较大迭代次数以减小误差
constexpr int try_count = 1024;

using gaussian_func_t = void(float *dest, float *src, int image_h, int image_w,
                             float sigma);
int main() {
  // 使用随机数据进行测试
  std::random_device rd;
  std::mt19937 gen(rd());
  std::ofstream test_result_aligned{"result.csv"};
  float *random_image = new float[w * h];
  float *out_buffer = new float[w * h];
  float *out_buffer2 = new float[w * h];
  float *out_buffer_cv2_gpu = new float[w * h];
  float *out_buffer_cv2_simd = new float[w * h];
  // 生成随机数据
  for (int i = 0; i < w * h; i++) {
    random_image[i] = std::generate_canonical<float, 10>(gen);
  }
  constexpr float sigma = 1.6;
  // warm up,移除CUDA启动开销
  gaussian_gpu(out_buffer, random_image, 1024, 1024, sigma);
  gaussian_cv2_cuda(out_buffer, random_image, 1024, 1024, sigma);

  // 测试不同实现的性能
  auto test_func = [&](int size, gaussian_func_t f, float *out,
                       const char *name) {
    std::chrono::duration<double> time_sum{0.0};
    for (int i = 0; i < try_count; i++) {
      const auto start_time{std::chrono::steady_clock::now()};
      f(out, random_image, size, size, sigma);
      const auto end_time{std::chrono::steady_clock::now()};
      const auto duration = end_time - start_time;
      time_sum += duration;
    }
    std::cout << name << "(" << try_count << "run): " << time_sum.count()
              << "sec" << std::endl;
    return time_sum;
  };
  // 误差计算函数
  auto err_func = [&](float *out1, float *out2, int size) {
    float err2 = 0;
    float sum = 0;
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        // GPU和CPU相加顺序不同（CPU顺序，GPU不一定），且CPU可能使用不同的浮点运算指令，单个比较不会一样
        err2 += (out1[i * size + j] - out2[i * size + j]) *
                (out1[i * size + j] - out2[i * size + j]);
        sum += out1[i * size + j] * out2[i * size + j];
      }
    }
    return sqrtf(err2 / sum);
  };
  auto tester = [&](int step, std::ofstream &result) {
    result << "size, cpu_time, cpu_simd_time, gpu_cv_time, gpu_time" << std::endl;
    // 测试不同的图像大小
    for (int size = 256; size <= w; size += step) {
      printf("size=%d, sigma=%f\n", size, sigma);
      // cpu
      auto cpu_time = test_func(size, gaussian_cpu, out_buffer, "cpu");
      // cpu simd opencv
      auto cpu_simd_time =
          test_func(size, gaussian_cv2_simd, out_buffer_cv2_simd, "cpu_simd_cv2");
      // gpu opencv
      auto gpu_cv_time =
          test_func(size, gaussian_cv2_cuda, out_buffer_cv2_gpu, "gpu_cv2");
      // gpu
      auto gpu_time = test_func(size, gaussian_gpu, out_buffer2, "gpu");

      // 对比CPU参考实现和GPU计算结果
      // 计算平均误差
      std::cout << "err: " << err_func(out_buffer, out_buffer2, size) << std::endl;
      std::cout << "err_cv2_simd: " << err_func(out_buffer, out_buffer_cv2_simd, size) << std::endl;
      std::cout << "err_cv2_cuda: " << err_func(out_buffer, out_buffer_cv2_gpu, size) << std::endl; 

      // 结果写入csv文件用于作图
      result << size << ", " << cpu_time.count() << ", "
             << cpu_simd_time.count() << ", " << gpu_cv_time.count() << ", "
             << gpu_time.count()  << std::endl;
    }
  };
  // 使用对齐的图片长宽
  tester(256, test_result_aligned);
  // 使用未对齐的图片长宽，control divergent影响
  // tester(253, test_result_unaligned);
  delete[] random_image;
  delete[] out_buffer;
  delete[] out_buffer2;
  delete[] out_buffer_cv2_gpu;
  delete[] out_buffer_cv2_simd;
  test_result_aligned.close();
  return 0;
}
