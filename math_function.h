#ifndef MATH_FUNCTION_H_
 #define MATH_FUNCTION_H_


// CUDA: thread number configuration.
// Use 1024 threads per block, which requires cuda sm_2x or above,
// or fall back to attempt compatibility (best of luck to you).
const int EXP_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int EXP_GET_BLOCKS(const int N) {
  return (N + EXP_CUDA_NUM_THREADS - 1) / EXP_CUDA_NUM_THREADS;
}


// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)



float cudnnConvOpt(float *A, float *B, float *C, int batch_size,
       int in_channels, int in_height, int in_width, int out_channels,
       int kernel_size, int stride, int pad);

float cudnnConv(float *A, float *B, float *C, int batch_size,
       int in_channels, int in_height, int in_width, int out_channels,
       int kernel_size, int stride, int pad);

float cudnnConv_algo(float *A, float *B, float *C, int batch_size,
      int in_channels, int in_height, int in_width, int out_channels,
      int kernel_size, int stride, int pad, int cudnn_algo);

float cublasConv(float *A, float *B, float *C, int batch_size,
      int in_channels, int in_height, int in_width, int out_channels,
      int kernel_size, int stride, int pad);

float csrConv(float *A, float *B, float *C, int batch_size,
      int in_channels, int in_height, int in_width, int out_channels,
      int kernel_size, int stride, int pad);

float im2col_gpu(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_col);

float im2col_gpu_batch(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_col, const int batch);

float cublasGEMM(bool TransA, bool TransB, float *A, float *B, float *C, int M,
  int N, int K);

float csrGEMM2(bool TransA, bool TransB, float *A, float *B, float *C, int M,
  int N, int K);

float cublas_transpose(const float *input, float *output, int in_M, int in_N);

void convert_dense2csr(const int M, const int N, const float *A, void **csrVal,
  void **csrRowPtr, void **csrColIndex, int *nnz);

float host_cublasGEMV(bool TransA, float *A, float *x, float *y, int M, int N);

float host_csrGEMV(bool TransA, float *A, float *x, float *y, int M, int N);

float cublasGEMV(bool TransA, float *A, float *x, float *y, int M, int N);

float csrGEMV(bool TransA, float *A, float *x, float *y, int M, int N);
#endif
