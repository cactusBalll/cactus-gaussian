#pragma once
#include<vector>
#include<cmath>

/// @brief 卷积核半径
constexpr int RADIUS = 8;
/// @brief 卷积核大小
constexpr int KERNEL_SIZE = 2 * RADIUS + 1;
/// @brief 获取1维高斯卷积核
/// @param sigma σ参数
/// @return 卷积核
static std::vector<float> make_gaussian_kernel_1d(float sigma) {
	auto ret = std::vector<float>(KERNEL_SIZE);
	const float PI = atan2(1.0, 0.0) * 2;
	auto g = [&](float x) {
		auto left = 1.0 / (sigma * sqrtf(2 * PI));
		auto right = expf(-(x * x) / (2 * sigma * sigma));
		return left * right;
		};
	float sum = 0;
	for (int i = -RADIUS; i <= RADIUS; i++) {
		float t = g(static_cast<float>(i));
		ret[i + RADIUS] = t;
		sum += t;
	}
	// 归一化
	for (int i = -RADIUS; i <= RADIUS; i++) {
		ret[i + RADIUS] /= sum;
	}
	return ret;
}