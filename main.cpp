#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>

#include "gaussian.h"

constexpr int w = 4096;
constexpr int h = 4096;
/// @brief  速度测试尝试求平均次数
constexpr int try_count = 24;
int main() {
  // 使用随机数据进行测试
  std::random_device rd;
  std::mt19937 gen(rd());
  std::ofstream test_result_aligned{"result.csv"};
  std::ofstream test_result_unaligned{"result_una.csv"};
  float *random_image = new float[w * h];
  float *out_buffer = new float[w * h];
  float *out_buffer2 = new float[w * h];
  // 生成随机数据
  for (int i = 0; i < w * h; i++) {
    random_image[i] = std::generate_canonical<float, 10>(gen);
  }
  constexpr float sigma = 1.6;
  // warm up,移除CUDA启动开销
  gaussian_gpu(out_buffer, random_image, 1024, 1024, sigma);

  auto tester = [&](int step, std::ofstream& result) {
    result << "size, cpu_time, gpu_time, err" << std::endl;
    // 测试不同的图像大小
    for (int size = 256; size <= w; size += step) {
      printf("size=%d, sigma=%f\n", size, sigma);
      // cpu
      std::chrono::duration<double> time_sum{0.0};
      for (int i = 0; i < try_count; i++) {
        const auto start_time{std::chrono::steady_clock::now()};
        gaussian_cpu(out_buffer, random_image, size, size, sigma);
        const auto end_time{std::chrono::steady_clock::now()};
        const auto duration = end_time - start_time;
        time_sum += duration;
      }
      std::cout << "CPU(" << try_count << "run): " << time_sum.count() << "sec"
                << std::endl;
      // gpu
      // 计时归零
      auto cpu_time = time_sum;
      time_sum = time_sum.zero();
      for (int i = 0; i < try_count; i++) {
        const auto start_time{std::chrono::steady_clock::now()};
        gaussian_gpu(out_buffer2, random_image, size, size, sigma);
        const auto end_time{std::chrono::steady_clock::now()};
        const auto duration = end_time - start_time;
        time_sum += duration;
      }
      std::cout << "GPU(" << try_count << "run): " << time_sum.count() << "sec"
                << std::endl;

      // 对比CPU参考实现和GPU计算结果
      // 计算平均误差
      float err2 = 0;
      float sum = 0;
      for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
          // GPU和CPU相加顺序不同（CPU顺序，GPU不一定），且CPU可能使用不同的浮点运算指令，单个比较不会一样
          // std::cout << out_buffer[i * size + j] << "    " <<
          // out_buffer2[i * size + j] << std::endl; if (fabsf(out_buffer[i
          // * size + j] - out_buffer2[i * size + j]) > 1e-4) {
          // std::cout << "different result!"
          // << std::endl; 	goto clean;
          // }
          err2 += (out_buffer[i * size + j] - out_buffer2[i * size + j]) *
                  (out_buffer[i * size + j] - out_buffer2[i * size + j]);
          sum += out_buffer[i * size + j] * out_buffer[i * size + j];
        }
      }
      std::cout << "err: " << sqrtf(err2 / sum) << std::endl;
      result << size << ", " << cpu_time.count() << ", " << time_sum.count() << ", " << sqrtf(err2 / sum) << std::endl;
    }
  };
  // 使用对齐的图片长宽
  tester(256, test_result_aligned);
  // 使用未对齐的图片长宽，control divergent影响
  tester(253, test_result_unaligned);
  delete[] random_image;
  delete[] out_buffer;
  test_result_aligned.close();
  test_result_unaligned.close();
  return 0;
}