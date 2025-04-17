#include<iostream>
#include<ctime>
#include<chrono>
#include<random>

#include "gaussian.h"

constexpr int w = 4096;
constexpr int h = 4096;
/// @brief  速度测试尝试求平均次数
constexpr int try_count = 16;
int main() {
	// 使用随机数据进行测试
	std::random_device rd;
    std::mt19937 gen(rd());
	float *random_image = new float[w * h];
	float *out_buffer = new float[w * h];
	// 生成随机数据
	for (int i = 0; i < w * h; i++) {
		random_image[i] = std::generate_canonical<float, 10>(gen);
	}
	constexpr float sigma = 1.6;
	// warm up,移除CUDA启动开销
	gaussian_gpu(out_buffer, random_image, 1024, 1024, sigma);
	// 测试不同的图像大小
	for (int size = 256; size <= w; size += 256) {
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
		std::cout<<"CPU(" << try_count << "run): " <<  time_sum.count() << "sec"<< std::endl;
		// gpu
		// 计时归零
		time_sum = time_sum.zero();
		for (int i = 0; i < try_count; i++) {
			const auto start_time{std::chrono::steady_clock::now()};
			gaussian_gpu(out_buffer, random_image, size, size, sigma);
			const auto end_time{std::chrono::steady_clock::now()};
			const auto duration = end_time - start_time;
			time_sum += duration;
		}
		std::cout<<"GPU(" << try_count << "run): " <<  time_sum.count() << "sec"<< std::endl;
	}


	delete[] random_image;
	delete[] out_buffer;
	return 0;
}