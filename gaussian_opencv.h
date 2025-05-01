#pragma once

extern "C" {

void gaussian_cv2_simd(float *dest, float *src, int image_h, int image_w,
                       float sigma);
void gaussian_cv2_cuda(float *dest, float *src, int image_h, int image_w,
                       float sigma);
}