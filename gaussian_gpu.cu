/// compute capabilities 以RTX4050为例，它支持8.9，则每个SM最多24个block
/// 24(block) * 64(thread) = 1536(SM)
/// 1536个thread
/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability
#include "common.h"
#include "gaussian.h"
#include <cassert>
#include <cooperative_groups.h>

/// 提供了更丰富的同步原语
/// 见https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups
namespace cg = cooperative_groups;

/// @brief 卷积核 存储在常量存储区
__constant__ float kernel[KERNEL_SIZE];

bool check_err() {
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("cuda error.\n");
    printf("err code: %d\n", static_cast<int>(err));
    printf("description: %s\n", cudaGetErrorString(err));
    return false;
  }
  return true;
}

void set_kernel(const float *k) {
  cudaMemcpyToSymbol(kernel, k, KERNEL_SIZE * sizeof(float));
}
/// @brief 行方向线程块大小x
constexpr int ROW_BLOCK_DIM_X = 16;
/// @brief 行方向线程块大小y
constexpr int ROW_BLOCK_DIM_Y = 4;
/// @brief 每个线程在每行生成卷积结果数量
constexpr int ROW_RESULT_STEP = 8;
/// @brief 每个线程两侧分别需要加载的光环元素
constexpr int ROW_HALO_STEP = 1;
/// @brief 处理的“砖块”大小x
constexpr int ROW_TILE_DIM_X = ROW_RESULT_STEP * ROW_BLOCK_DIM_X;
/// @brief 处理的“砖块”大小y
constexpr int ROW_TILE_DIM_Y = ROW_BLOCK_DIM_Y;
/// @brief shared memory行大小
constexpr int ROW_LOAD_SIZE_X =
    (ROW_RESULT_STEP + 2 * ROW_HALO_STEP) * ROW_BLOCK_DIM_X;

__global__ void convolution_row_kernel(float *dest, float *src, int image_h,
                                       int image_w) {
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float shared_data[ROW_BLOCK_DIM_Y][ROW_LOAD_SIZE_X];

  // 计算TILE起始位置偏移，x方向减去一个HALO_STEP距离，光圈元素
  // base_x已经加上了threadIdx.x的偏移
  const int base_x =
      (blockIdx.x * ROW_RESULT_STEP - ROW_HALO_STEP) * ROW_BLOCK_DIM_X +
      threadIdx.x;
  const int base_y = blockIdx.y * ROW_BLOCK_DIM_Y + threadIdx.y;

  // 直接修改形参，应用偏移
  dest += base_y * image_w + base_x;
  src += base_y * image_w + base_x;

// 加载数据
// 编译指令：循环展开
#pragma unroll
  for (int i = 0; i <= ROW_HALO_STEP + ROW_RESULT_STEP; i++) {
    // 如果不假定图片长宽被TILE整除，则中间也需要判断幽灵元素，base_y一定大于等于0
    if (base_x + i * ROW_BLOCK_DIM_X >= 0 &&
        base_x + i * ROW_BLOCK_DIM_X < image_w && base_y < image_h) {
      shared_data[threadIdx.y][threadIdx.x + i * ROW_BLOCK_DIM_X] =
          src[i * ROW_BLOCK_DIM_X];
    } else {
      shared_data[threadIdx.y][threadIdx.x + i * ROW_BLOCK_DIM_X] = 0;
    }
  }
  // 同步，相当于__syncthreads
  cg::sync(cta);
#pragma unroll
  for (int i = ROW_HALO_STEP; i < ROW_HALO_STEP + ROW_RESULT_STEP; i++) {
    // 执行卷积，每个thread计算了多个位置
    float sum = 0;
#pragma unroll
    for (int j = -RADIUS; j <= RADIUS; j++) {
      sum += kernel[RADIUS + j] *
             shared_data[threadIdx.y][threadIdx.x + i * ROW_BLOCK_DIM_X + j];
    }
    // 在范围内则写入
    if (base_x + i * ROW_BLOCK_DIM_X >= 0 &&
        base_x + i * ROW_BLOCK_DIM_X < image_w && base_y < image_h) {
      dest[i * ROW_BLOCK_DIM_X] = sum;
    }
  }
}

constexpr int COLUMN_BLOCK_DIM_X = 16;
constexpr int COLUMN_BLOCK_DIM_Y = 8;
constexpr int COLUMN_RESULT_STEP = 8;
constexpr int COLUMN_HALO_STEP = 1;
constexpr int COLUMN_TILE_DIM_X = COLUMN_BLOCK_DIM_X;
constexpr int COLUMN_TILE_DIM_Y = COLUMN_BLOCK_DIM_Y * COLUMN_RESULT_STEP;
constexpr int COLUMN_LOAD_SIZE_Y =
    (COLUMN_RESULT_STEP + COLUMN_HALO_STEP * 2) * COLUMN_BLOCK_DIM_Y;
__global__ void convolution_column_kernel(float *dest, float *src, int image_h,
                                          int image_w) {
  cg::thread_block cta = cg::this_thread_block();
  // 这里是转置的
  __shared__ float shared_data[COLUMN_TILE_DIM_X][COLUMN_LOAD_SIZE_Y];
  // threadIdx.x递增+1 接合
  const int base_x = blockIdx.x * COLUMN_BLOCK_DIM_X + threadIdx.x;
  const int base_y = (blockIdx.y * COLUMN_RESULT_STEP - COLUMN_HALO_STEP) *
                         COLUMN_BLOCK_DIM_Y +
                     threadIdx.y;
  dest += base_y * image_w + base_x;
  src += base_y * image_w + base_x;

// 加载元素，依次上方光环，主要，下方光环
#pragma unroll
  for (int i = 0; i <= COLUMN_HALO_STEP + COLUMN_RESULT_STEP; i++) {
    if (base_y + i * COLUMN_BLOCK_DIM_Y >= 0 &&
        base_y + i * COLUMN_BLOCK_DIM_Y < image_h && base_x < image_w) {
      shared_data[threadIdx.x][threadIdx.y + i * COLUMN_BLOCK_DIM_Y] =
          src[i * COLUMN_BLOCK_DIM_Y * image_w];
    } else {
      shared_data[threadIdx.x][threadIdx.y + i * COLUMN_BLOCK_DIM_Y] = 0;
    }
  }
  cg::sync(cta);
// 在列上执行卷积
#pragma unroll
  for (int i = COLUMN_HALO_STEP; i < COLUMN_HALO_STEP + COLUMN_RESULT_STEP;
       i++) {
    float sum = 0;
#pragma unroll
    for (int j = -RADIUS; j <= RADIUS; j++) {
      sum += kernel[j + RADIUS] *
             shared_data[threadIdx.x][threadIdx.y + i * COLUMN_BLOCK_DIM_Y + j];
    }
    // 在范围内则写回
    if (base_y + i * COLUMN_BLOCK_DIM_Y >= 0 &&
        base_y + i * COLUMN_BLOCK_DIM_Y < image_h && base_x < image_w) {
      dest[i * COLUMN_BLOCK_DIM_Y * image_w] = sum;
    }
  }
}

void convolution_row_gpu(float *dest, float *src, int image_h, int image_w) {
  // tile必须覆盖整个区域，不整除则加1
  int bx = image_w / ROW_TILE_DIM_X;
  if (image_w % ROW_TILE_DIM_X) {
    bx += 1;
  }
  int by = image_h / ROW_TILE_DIM_Y;
  if (image_h % ROW_TILE_DIM_Y) {
    by += 1;
  }
  dim3 blocks_row(bx, by);
  dim3 threads_row(ROW_BLOCK_DIM_X, ROW_BLOCK_DIM_Y);
  convolution_row_kernel<<<blocks_row, threads_row>>>(dest, src, image_h,
                                                      image_w);
}
void convolution_column_gpu(float *dest, float *src, int image_h, int image_w) {
  // tile必须覆盖整个区域，不整除则加1
  int bx = image_w / COLUMN_TILE_DIM_X;
  if (image_w % COLUMN_TILE_DIM_X) {
    bx += 1;
  }
  int by = image_h / COLUMN_TILE_DIM_Y;
  if (image_h % COLUMN_TILE_DIM_Y) {
    by += 1;
  }
  dim3 blocks_column(bx, by);
  dim3 threads_column(COLUMN_BLOCK_DIM_X, COLUMN_BLOCK_DIM_Y);
  convolution_column_kernel<<<blocks_column, threads_column>>>(
      dest, src, image_h, image_w);
}
void gaussian_gpu(float *dest, float *src, int image_h, int image_w,
                  float sigma) {
  // 至少1个tile大
  assert(image_h > COLUMN_TILE_DIM_Y);
  assert(image_w > ROW_TILE_DIM_X);
  // 且必须是TILE的整数倍
  // 对于照片这通常是满足的如3072*4096
  // assert(image_h % COLUMN_TILE_DIM_Y == 0);
  // assert(image_w % ROW_TILE_DIM_Y == 0);

  const int buffer_size = image_h * image_w * sizeof(float);

  // 设备上内存指针
  float *input_device = nullptr;
  float *med_device = nullptr;
  float *output_device = nullptr;

  cudaMalloc(&input_device, buffer_size);
  cudaMalloc(&med_device, buffer_size);
  cudaMalloc(&output_device, buffer_size);
  // 检查错误
  if (!check_err()) {
    cudaFree(input_device);
    cudaFree(med_device);
    cudaFree(output_device);
    return;
  }
  // 从内存复制输入到设备
  cudaMemcpy(input_device, src, buffer_size, cudaMemcpyHostToDevice);
  // 检查错误
  if (!check_err()) {
    cudaFree(input_device);
    cudaFree(med_device);
    cudaFree(output_device);
    return;
  }

  auto kernel_vec = make_gaussian_kernel_1d(sigma);
  // 高斯卷积核是对称的，分离的两个1维卷积使用同样的卷积核
  set_kernel(kernel_vec.data());
  convolution_row_gpu(med_device, input_device, image_h, image_w);
  convolution_column_gpu(output_device, med_device, image_h, image_w);

  // 从设备把结果复制回内存
  cudaMemcpy(dest, output_device, buffer_size, cudaMemcpyDeviceToHost);
  check_err();
  cudaFree(input_device);
  cudaFree(med_device);
  cudaFree(output_device);
}