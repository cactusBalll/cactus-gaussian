#pragma once

// comments encoding: UTF-8
extern "C" {
	/// @brief 行上一维高斯卷积
	/// @param dest 输出f32灰度图片
	/// @param src 输入f32灰度图片
	/// @param image_h 图片高度
	/// @param image_w 图片宽度
	/// @param kernel 卷积核
	void convolution_row_cpu(float* dest, float* src, int image_h, int image_w, const float* kernel);
	/// @brief 列上一维高斯卷积
	/// @param dest 输出f32灰度图片
	/// @param src 输入f32灰度图片
	/// @param image_h 图片高度
	/// @param image_w 图片宽度
	/// @param kernel 卷积核
	void convolution_column_cpu(float* dest, float* src, int image_h, int image_w, const float* kernel);
	/// @brief 高斯模糊CPU
	/// @param dest 输出f32灰度图片
	/// @param src 输入f32灰度图片
	/// @param image_h 图片高度
	/// @param image_w 图片宽度
	/// @param sigma σ参数，标准差
	void gaussian_cpu(float* dest, float* src, int image_h, int image_w, float sigma);

	/// @brief 将卷积核设置到GPU常量缓存
	/// @param k 卷积核
	void set_kernel(const float* k);

	/// @brief 行上一维卷积
	/// @param dest 输出f32灰度图片
	/// @param src 输入f32灰度图片
	/// @param image_h 图片高度
	/// @param image_w 图片宽度
	void convolution_row_gpu(float* dest, float* src, int image_h, int image_w);
	/// @brief 列上一维卷积
	/// @param dest 输出f32灰度图片
	/// @param src 输入f32灰度图片
	/// @param image_h 图片高度
	/// @param image_w 图片宽度
	void convolution_column_gpu(float* dest, float* src, int image_h, int image_w);
	/// @brief 高斯模糊GPU
	/// @param dest 输出f32灰度图片
	/// @param src 输入f32灰度图片
	/// @param image_h 图片高度
	/// @param image_w 图片宽度
	/// @param sigma σ参数，标准差
	void gaussian_gpu(float* dest, float* src, int image_h, int image_w, float sigma);
	
}