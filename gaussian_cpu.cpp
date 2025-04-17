#include "gaussian.h"
#include "common.h"

void convolution_row_cpu(float* dest, float* src, int image_h, int image_w, const float* kernel) {
	for (int y = 0; y < image_h; y++) {
		for (int x = 0; x < image_w; x++) {
			float sum = 0;
			for (int r = -RADIUS; r <= RADIUS; r++) {
				int i = x + r;
				if (i >= 0 && i <= image_w) {
					sum += src[y * image_w + i] * kernel[r + RADIUS];
				}
			}
			dest[y * image_w + x] = sum;
		}
	}
}
void convolution_column_cpu(float* dest, float* src, int image_h, int image_w, const float* kernel) {
	float* temp = new float[image_h * image_w];
	float* temp2 = new float[image_h * image_w];
	// 转置，避免按列访问cache不友好
	for (int i = 0; i < image_w; i++) {
		for (int j = 0; j < image_h; j++) {
			temp[i * image_h + j] = src[j * image_w + i];
		}
	}
	// 与行方向卷积方法相同，交换宽高
	convolution_row_cpu(temp2, temp, image_w, image_h, kernel);

	// 转置，得到结果
	for (int i = 0; i < image_h; i++) {
		for (int j = 0; j < image_w; j++) {
			dest[i * image_w + j] = temp2[j * image_h + i];
		}
	}

	delete[] temp;
	delete[] temp2;
}
void gaussian_cpu(float* dest, float* src, int image_h, int image_w, float sigma) {
	auto kernel = make_gaussian_kernel_1d(sigma);
	float* temp = new float[image_h * image_w];
	// 两次一维卷积得到二维高斯核卷积的结果
	convolution_row_cpu(temp, src, image_h, image_w, kernel.data());
	convolution_column_cpu(dest, temp, image_h, image_w, kernel.data());
	delete[] temp;
}