#include "math_function.h"

#include <cusparse.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <sys/time.h>

#include "common.h"   // for check utils
#include "common_cuda.h"

float cudnnConvOpt(float *A, float *B, float *C, int batch_size,
      int in_channels, int in_height, int in_width, int out_channels,
      int kernel_size, int stride, int pad) {

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float total_time = 0.0;

  int out_width = (in_width + 2 * pad - kernel_size) / stride + 1;
  int out_height = (in_height + 2 * pad - kernel_size) / stride + 1;

  cudnnHandle_t handle = 0;
  CUDNN_CHECK(cudnnCreate(&handle));
  cudnnDataType_t float_type = CUDNN_DATA_FLOAT;

  // workspace
  size_t workspace_fwd_size = 0;
  void *workspaceData = NULL;  // underlying storage

  // algorithms for forward convolutions
  cudnnConvolutionFwdAlgo_t fwd_algo;

  // descriptors
  cudnnTensorDescriptor_t    bottom_desc, top_desc;
  cudnnFilterDescriptor_t    filter_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  // Create filter descriptor
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, float_type,
    CUDNN_TENSOR_NCHW,
    out_channels, in_channels, kernel_size, kernel_size));

  // Create tensor descriptor for data and convolutions
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&bottom_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&top_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

  // initializes the previously created generic Tensor descriptor object into a
  // 4D tensor
  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(bottom_desc, float_type,
        batch_size, in_channels, in_height, in_width,
        in_channels * in_height * in_width,
        in_height * in_width,
        in_width, 1));

  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(top_desc, float_type,
        batch_size, out_channels, out_height, out_width,
        out_channels * out_height * out_width,
        out_height * out_width,
        out_width, 1));

  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
      pad, pad, stride, stride, 1, 1,
      CUDNN_CROSS_CORRELATION, float_type));
      // CUDNN_CONVOLUTION, float_type));


  // Now try to start the cuDNN process
  size_t workspace_limit_bytes, total_memory;
  CUDA_CHECK(cudaMemGetInfo(&workspace_limit_bytes, &total_memory));

  int returned_algo_cnt = 0;
  cudnnConvolutionFwdAlgoPerf_t fwd_algo_perf;
  CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(handle,
  bottom_desc,
  filter_desc,
  conv_desc,
  top_desc,
  1,
  &returned_algo_cnt,
  &fwd_algo_perf));

  LOG("fwd_algo: %d", fwd_algo_perf.algo);
  CUDA_CHECK(cudaMalloc((void **)&workspaceData, fwd_algo_perf.memory));
  float oneval = 1.0, zeroval = 0.0;
  void *one = (void *)&oneval;
  void *zero = (void *)&zeroval;

  cudaDeviceSynchronize();

  cudaEventRecord(start, 0);
  CUDNN_CHECK(cudnnConvolutionForward(handle,
                           one,
                           bottom_desc,
                           A,
                           filter_desc,
                           B,
                           conv_desc,
                           //fwd_algo,
                           fwd_algo_perf.algo,
                           workspaceData,
                           //workspace_fwd_size,
                           fwd_algo_perf.memory,
                           zero,
                           top_desc,
                           C));
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&total_time, start, stop);

  LOG("cuDNN convolution with NCHW format Time: %f ms", total_time);

  CUDA_CHECK(cudaFree(workspaceData));

  return total_time;
}

float cudnnConv(float *A, float *B, float *C, int batch_size,
      int in_channels, int in_height, int in_width, int out_channels,
      int kernel_size, int stride, int pad) {

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float total_time = 0.0;

  int out_width = (in_width + 2 * pad - kernel_size) / stride + 1;
  int out_height = (in_height + 2 * pad - kernel_size) / stride + 1;

  cudnnHandle_t handle = 0;
  CUDNN_CHECK(cudnnCreate(&handle));
  cudnnDataType_t float_type = CUDNN_DATA_FLOAT;

  // workspace
  size_t workspace_fwd_size = 0;
  void *workspaceData = NULL;  // underlying storage

  // algorithms for forward convolutions
  cudnnConvolutionFwdAlgo_t fwd_algo;

  // descriptors
  cudnnTensorDescriptor_t    bottom_desc, top_desc;
  cudnnFilterDescriptor_t    filter_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  // Create filter descriptor
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, float_type,
    CUDNN_TENSOR_NCHW,
    out_channels, in_channels, kernel_size, kernel_size));

  // Create tensor descriptor for data and convolutions
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&bottom_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&top_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

  // initializes the previously created generic Tensor descriptor object into a
  // 4D tensor
  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(bottom_desc, float_type,
        batch_size, in_channels, in_height, in_width,
        in_channels * in_height * in_width,
        in_height * in_width,
        in_width, 1));

  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(top_desc, float_type,
        batch_size, out_channels, out_height, out_width,
        out_channels * out_height * out_width,
        out_height * out_width,
        out_width, 1));

  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
      pad, pad, stride, stride, 1, 1,
      CUDNN_CROSS_CORRELATION, float_type));
      // CUDNN_CONVOLUTION, float_type));


  // Now try to start the cuDNN process
  size_t workspace_limit_bytes, total_memory;
  CUDA_CHECK(cudaMemGetInfo(&workspace_limit_bytes, &total_memory));

  int returned_algo_cnt = 0;
  cudnnConvolutionFwdAlgoPerf_t fwd_algo_perf;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(handle,
  bottom_desc,
  filter_desc,
  conv_desc,
  top_desc,
  1,
  &returned_algo_cnt,
  &fwd_algo_perf));

  LOG("fwd_algo: %d", fwd_algo_perf.algo);

  // allocate workspace
  CUDA_CHECK(cudaMalloc((void **)&workspaceData, fwd_algo_perf.memory));
  float oneval = 1.0, zeroval = 0.0;
  void *one = (void *)&oneval;
  void *zero = (void *)&zeroval;

  cudaDeviceSynchronize();

  cudaEventRecord(start, 0);
  CUDNN_CHECK(cudnnConvolutionForward(handle,
                           one,
                           bottom_desc,
                           A,
                           filter_desc,
                           B,
                           conv_desc,
                           fwd_algo_perf.algo,
                           workspaceData,
                           fwd_algo_perf.memory,
                           zero,
                           top_desc,
                           C));
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&total_time, start, stop);

  LOG("cuDNN convolution with NCHW format Time: %f ms", total_time);

  CUDA_CHECK(cudaFree(workspaceData));

  return total_time;
}

float cudnnConv_algo(float *A, float *B, float *C, int batch_size,
      int in_channels, int in_height, int in_width, int out_channels,
      int kernel_size, int stride, int pad, int cudnn_algo) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float total_time = 0.0;

  int out_width = (in_width + 2 * pad - kernel_size) / stride + 1;
  int out_height = (in_height + 2 * pad - kernel_size) / stride + 1;

  // cudnn_handle;
  cudnnHandle_t handle = 0;
  CUDNN_CHECK(cudnnCreate(&handle));
  cudnnDataType_t float_type = CUDNN_DATA_FLOAT;

  // workspace
  size_t workspace_fwd_size = 0;
  void *workspaceData = NULL;  // underlying storage

  // algorithms for forward convolutions
  cudnnConvolutionFwdAlgo_t fwd_algo;

  // descriptors
  cudnnTensorDescriptor_t    bottom_desc, top_desc;
  cudnnFilterDescriptor_t    filter_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  // Create filter descriptor
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, float_type,
    CUDNN_TENSOR_NCHW,
    out_channels, in_channels, kernel_size, kernel_size));

  // Create tensor descriptor for data and convolutions
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&bottom_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&top_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

  // initializes the previously created generic Tensor descriptor object into a
  // 4D tensor
  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(bottom_desc, float_type,
        batch_size, in_channels, in_height, in_width,
        in_channels * in_height * in_width,
        in_height * in_width,
        in_width, 1));

  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(top_desc, float_type,
        batch_size, out_channels, out_height, out_width,
        out_channels * out_height * out_width,
        out_height * out_width,
        out_width, 1));

  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
      pad, pad, stride, stride, 1, 1,
      CUDNN_CROSS_CORRELATION, float_type
      // CUDNN_CONVOLUTION, float_type
    ));

  fwd_algo = (cudnnConvolutionFwdAlgo_t)cudnn_algo;

  LOG("fwd_algo: %d", fwd_algo);

  // get workspace size
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle,
      bottom_desc,
      filter_desc,
      conv_desc,
      top_desc,
      fwd_algo,
      &workspace_fwd_size));

  // allocate workspace
  CUDA_CHECK(cudaMalloc((void **)&workspaceData, workspace_fwd_size));

  float oneval = 1.0, zeroval = 0.0;
  void *one = (void *)&oneval;
  void *zero = (void *)&zeroval;

  cudaDeviceSynchronize();

  cudaEventRecord(start, 0);
  CUDNN_CHECK(cudnnConvolutionForward(handle,
                           one,
                           bottom_desc,
                           A,
                           filter_desc,
                           B,
                           conv_desc,
                           fwd_algo, workspaceData,
                           workspace_fwd_size,
                           zero,
                           top_desc,
                           C));
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&total_time, start, stop);

  LOG("cuDNN convolution with NCHW format Time: %f ms", total_time);

  CUDA_CHECK(cudaFree(workspaceData));

  return total_time;
}

float cublasConv(float *A, float *B, float *C, int batch_size,
      int in_channels, int in_height, int in_width, int out_channels,
      int kernel_size, int stride, int pad) {

  float total_time = 0.0;
  // struct timeval start, stop;

  int out_width = (in_width + 2 * pad - kernel_size) / stride + 1;
  int out_height = (in_height + 2 * pad - kernel_size) / stride + 1;

  // For each image in the batch, do convolution separately
  float *input = NULL;
  float *weights = B;
  float *output = NULL;
  float *unroll_buff = NULL; // column buffer

  float *unroll_matrix = NULL;
  size_t unroll_matrix_size = sizeof(float) *
    batch_size * in_channels * kernel_size * kernel_size * out_width * out_height;
  CUDA_CHECK(cudaMalloc((void **)&unroll_matrix, unroll_matrix_size));
  unroll_buff = A;
  if (kernel_size > 1) {
    total_time += im2col_gpu_batch(A, in_channels,
                          in_height, in_width,
                          kernel_size, kernel_size,
                          pad, pad,
                          stride, stride, unroll_matrix, batch_size);
    unroll_buff = unroll_matrix;
  }
  // M = output_channels, N = output_h * output_w * batch_size
  // K = input_channels * kernel_size * kernel_size
  total_time += cublasGEMM(false, false, weights, unroll_buff,
     C, out_channels, out_height * out_width * batch_size,
     in_channels * kernel_size * kernel_size);

  LOG("cublasConv total Time: %f ms", total_time);

  CUDA_CHECK(cudaFree(unroll_matrix));
  return total_time;
}

__global__ void im2col_gpu_kernel(const int n, const float* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    float* data_col) {
  // output also store in input_channels * r * s * (output_h * output_w) format
  // sequentially pull out index
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_col; // which output column
    int h_index = index / width_col;
    int h_out = h_index % height_col; // which output row
    int channel_in = h_index / height_col; // which channel
    int channel_out = channel_in * kernel_h * kernel_w;
    int h_in = h_out * stride_h - pad_h; // the start of the input region
    int w_in = w_out * stride_w - pad_w;
    float* data_col_ptr = data_col;
    data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    const float* data_im_ptr = data_im;
    data_im_ptr += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
            data_im_ptr[i * width + j] : (float)(0);
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

float im2col_gpu(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_col) {

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float total_time = 0.0;

  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;

  cudaDeviceSynchronize();
  cudaEventRecord(start, 0);
  // NOLINT_NEXT_LINE(whitespace/operators)

  im2col_gpu_kernel<<<EXP_GET_BLOCKS(num_kernels),
                             EXP_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, height_col,
      width_col, data_col);
  CUDA_POST_KERNEL_CHECK;

  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&total_time, start, stop);
  LOG("im2col_gpu total Time: %f ms", total_time);

  return total_time;
}

__global__ void batch_im2col_gpu_kernel(const int n, const float* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int batch, const int channels, const int height_col, const int width_col,
    float* data_col) {
  // output also store in input_channels * r * s * (output_h * output_w) format
  // sequentially pull out index
  CUDA_KERNEL_LOOP(id, n) {
    int index = id % (channels * height_col * width_col);
    int batch_id = id / (channels * height_col * width_col);
    int w_out = index % width_col; // which output column
    int h_index = index / width_col;
    int h_out = h_index % height_col; // which output row
    int channel_in = h_index / height_col; // which channel
    int channel_out = channel_in * kernel_h * kernel_w;
    int h_in = h_out * stride_h - pad_h; // the start of the input region
    int w_in = w_out * stride_w - pad_w;
    float* data_col_ptr = data_col;
    data_col_ptr += ((channel_out * batch + batch_id) * height_col + h_out)
                  * width_col + w_out;
    const float* data_im_ptr = data_im;
    data_im_ptr += ((batch_id * channels + channel_in) * height + h_in) * width
                  + w_in;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
            data_im_ptr[i * width + j] : (float)(0);
        data_col_ptr += height_col * width_col * batch;
      }
    }
  }
}

float im2col_gpu_batch(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_col, const int batch) {

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float total_time = 0.0;

  // We are going to launch channels * height_col * width_col * batch kernels,
  // each kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  num_kernels *= batch;

  cudaDeviceSynchronize();
  cudaEventRecord(start, 0);
  // NOLINT_NEXT_LINE(whitespace/operators)

  batch_im2col_gpu_kernel<<<EXP_GET_BLOCKS(num_kernels),
                             EXP_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w,
      batch, channels, height_col, width_col, data_col);
  CUDA_POST_KERNEL_CHECK;

  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&total_time, start, stop);
  LOG("im2col_gpu_batch total Time: %f ms", total_time);

  return total_time;
}

// GEMM with cuBLAS
// A size M*K, B size K*N
// if M == 1(which means A is a vector), falls into GEMV
float cublasGEMM(bool TransA, bool TransB, float *A, float *B, float *C,
    int M, int N, int K) {
  cublasHandle_t handle = NULL;
  CUBLAS_CHECK(cublasCreate(&handle));

  const float alpha = 1.0;
  const float beta = 0.0;

  int lda = (TransA == false) ? K : M;
  int ldb = (TransB == false) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == false) ? CUBLAS_OP_N : CUBLAS_OP_T;

  // struct timeval start, stop;
  // Timer timer;
  float total_time;

  cudaDeviceSynchronize();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // cudaEventRecord(start, 0);


  if (M == 1) {
    LOG("USE cublasSgemv!");
    cudaEventRecord(start, 0);
    // timer.start();
    // wrong execution actually...
    CUBLAS_CHECK(cublasSgemv(handle,
            cuTransB,

            K, // number of rows of matrix A, not op(A)!
            N,
            // N,
            // K,
            &alpha,
            B,
            ldb,
            A,
            1,
            &beta,
            C,
            1
            ));
    cudaDeviceSynchronize();
    // timer.end();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
  }
  else {
    cudaEventRecord(start, 0);
    // timer.start();
    // Note that cuBLAS use Fortran order (column-major)
    // But we use row-major, so we need to switch the A, B matrix
    CUBLAS_CHECK(cublasSgemm(handle,
                cuTransB,
                cuTransA,
                N,
                M,
                K,
                &alpha,
                B,
                ldb,
                A,
                lda,
                &beta,
                C,
                N
                ));

    cudaDeviceSynchronize();
    // timer.end();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
  }

  // total_time = timer.duration_ms();
  // LOG("cuBLAS Time: %f ms", total_time);
  cudaEventElapsedTime(&total_time, start, stop);
  LOG("cuBLAS total Time by cudaEvent: %f ms", total_time);

  return total_time;
}

// Do convolution with cuSparse CSR format
float csrConv(float *A, float *B, float *C, int batch_size,
      int in_channels, int in_height, int in_width, int out_channels,
      int kernel_size, int stride, int pad) {

  float total_time = 0.0;
  double im2col_time = 0.0, transpose_time = 0.0;
  double tmp_time = 0.0;

  int out_width = (in_width + 2 * pad - kernel_size) / stride + 1;
  int out_height = (in_height + 2 * pad - kernel_size) / stride + 1;

  // For each image in the batch, do convolution separately
  float *input = NULL;
  float *weights = B;
  float *unroll_buff = NULL; // column buffer

  float *unroll_matrix = NULL;
  size_t unroll_matrix_size = sizeof(float) *
    batch_size * in_channels * kernel_size * kernel_size * out_width * out_height;
  CUDA_CHECK(cudaMalloc((void **)&unroll_matrix, unroll_matrix_size));
  unroll_buff = A;
  if (kernel_size > 1) {
    tmp_time = im2col_gpu_batch(A, in_channels,
                          in_height, in_width,
                          kernel_size, kernel_size,
                          pad, pad,
                          stride, stride, unroll_matrix, batch_size);
    unroll_buff = unroll_matrix;
    total_time += tmp_time;
    im2col_time += tmp_time;
  }

  // we need to manually transpose the output
  float *output_trans;
  CUDA_CHECK(cudaMalloc((void **)&output_trans,
      sizeof(float) * batch_size * out_channels * out_height * out_width));


  int N = out_height * out_width * batch_size;
  int M = out_channels;
  int K = in_channels * kernel_size * kernel_size;
  total_time += csrGEMM2(true, true, unroll_buff, weights,
      output_trans, N, M, K);

  tmp_time = cublas_transpose(output_trans, C,
      out_height * out_width * batch_size, out_channels);
  transpose_time += tmp_time;


  LOG("csrConv im2col Time: %f, transpose time, %f, total Time: %f ms",
   im2col_time, transpose_time, total_time);

  CUDA_CHECK(cudaFree(unroll_matrix));
  CUDA_CHECK(cudaFree(output_trans));
  return total_time;
}

// GEMM with cuSparse CSR format for weights
// cusparseScsrmm
// Bt * At = (AB)t
// Remember that we have to manually transpose B...
// csrGEMM2 support At and Bt together
float csrGEMM2(bool TransA, bool TransB, float *A, float *B, float *C,
      int M, int N, int K) {

  cusparseHandle_t handle = 0;
  CUSPARSE_CHECK(cusparseCreate(&handle));
  cusparseMatDescr_t descr = 0;
  CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  int nnz;
  float *csrVal;
  int *csrRowPtr;
  int *csrColIndex;

  // Note the different of row-major (host) and column-major (device)!
  int B_row_dev = N;
  int B_col_dev = K;
  if (TransB) {
    float *Bt;
    CUDA_CHECK(cudaMalloc((void**)&Bt, sizeof(float) * N * K));
    cublas_transpose(B, Bt, N, K);
   //  printf("Have transposed B...\n");

    convert_dense2csr(K, N, Bt, (void **)&csrVal, (void **)&csrRowPtr,
    (void **)&csrColIndex, &nnz);

    CUDA_CHECK(cudaFree(Bt));
  } else {

    convert_dense2csr(K, N, B, (void **)&csrVal, (void **)&csrRowPtr,
    (void **)&csrColIndex, &nnz);
  }

  const float alpha = 1.0;
  const float beta = 0.0;

  int lda = (TransA == false) ? K : M;
  int A_col_dev = (TransA == false) ? M : K;
  int ldc = N;

  // int ldb = (TransB == false) ? N : K;
  cusparseOperation_t cuTransA =
      (TransA == false) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
  cusparseOperation_t cuTransB = CUSPARSE_OPERATION_NON_TRANSPOSE;


  float total_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaDeviceSynchronize();

  // struct timeval start, stop;

  // cudaDeviceSynchronize();

  if (M == 1) {
    LOG("USE csrSgemv!");
    // gettimeofday(&start, NULL);
    cudaEventRecord(start, 0);
    // wrong execution actually....
    CUSPARSE_CHECK(cusparseScsrmv(handle,
                cuTransB,
                B_row_dev,
                B_col_dev,
                nnz,
                &alpha,
                descr,
                csrVal,
                csrRowPtr,
                csrColIndex,
                A,
                &beta,
                C));

    // cudaDeviceSynchronize();
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&total_time, start, stop);
    // gettimeofday(&stop, NULL);
  }
  else {
    // gettimeofday(&start, NULL);
    cudaEventRecord(start, 0);
    // Note that cuBLAS use Fortran order (column-major)
    // But we use row-major, so we need to switch the A, B matrix
    CUSPARSE_CHECK(cusparseScsrmm2(handle,
                cuTransB,
                cuTransA,
                B_row_dev,
                M,
                B_col_dev,
                nnz,
                &alpha,
                descr,
                csrVal,
                csrRowPtr,
                csrColIndex,
                A,
                lda,
                &beta,
                C,
                ldc));

    // cudaDeviceSynchronize();
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // gettimeofday(&stop, NULL);
  }
  cudaEventElapsedTime(&total_time, start, stop);
  //
  // total_time = (stop.tv_sec - start.tv_sec) * 1000.0 +
  //              (stop.tv_usec - start.tv_usec) / 1000.0;
  LOG("cuSparse csrMM2 Time: %f ms", total_time);

  CUDA_CHECK(cudaFree(csrVal));
  CUDA_CHECK(cudaFree(csrRowPtr));
  CUDA_CHECK(cudaFree(csrColIndex));

  return total_time;
}

// transpose a matrix
// in_M and in_N are the input M, N
float cublas_transpose(const float *input, float *output,
      int in_M, int in_N)
{

  const float alpha = 1.0;
  const float beta = 0.0;

  cudaEvent_t start, stop;
  float total_time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaDeviceSynchronize();
  cudaEventRecord(start, 0);

  // cublas use column major
  // before trans: M * N (row-major on Host) / N * M (column-majore)
  // after trans: N * M on Host / M * N on device
  cublasHandle_t handle = NULL;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSgeam(handle,
                CUBLAS_OP_T,
                CUBLAS_OP_T,
                in_M,
                in_N,
                &alpha,
                input,
                in_N,
                &beta,
                input,
                in_N,
                output,
                in_M
                ));

  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&total_time, start, stop);

  return total_time;
}

// convert a matrix A into CSR format
// M, N are rows and columns on Host row-major!
// as a M*N row-major matrix is equivelent to a N*M column-major matrix,
// and the row-major matrix in CSR format is the same to the column-major matrix
// in CSC format,
// we treat the input M*N row-major matrix as N*M column-major matrix,
// use cusparse to compute this columnn-major matrix into CSC, and return it
// as a row-major CSR
void convert_dense2csr(const int M, const int N, const float *A,
  void **csrVal, void **csrRowPtr, void **csrColIndex, int *nnz) {

    cusparseHandle_t handle = 0;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    cusparseMatDescr_t descr = 0;
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  // Note that cusparse follows Fortran order (column-major)
  // So rows = N, columns = M
  int *nnzPerRowColumn;
  int lda = N;
  CUDA_CHECK(cudaMalloc((void**)&nnzPerRowColumn, sizeof(int) * M));

  cudaDeviceSynchronize();

  CUSPARSE_CHECK(cusparseSnnz(handle, CUSPARSE_DIRECTION_COLUMN, N, M, descr, A,
    lda, nnzPerRowColumn, nnz));

  cudaDeviceSynchronize();

  CUDA_CHECK(cudaMalloc(csrVal, sizeof(float) * (*nnz)));
  CUDA_CHECK(cudaMalloc(csrRowPtr, sizeof(int) * (M + 1)));
  CUDA_CHECK(cudaMalloc(csrColIndex, sizeof(int) * (*nnz)));

  // convert to CSC format with column major, which is equivelant to CSR format
  // with row major
  CUSPARSE_CHECK(cusparseSdense2csc(handle, N, M, descr, A, lda,
      nnzPerRowColumn, (float*)*csrVal, (int*)*csrColIndex, (int*)*csrRowPtr));

  cudaDeviceSynchronize();

  CUDA_CHECK(cudaFree(nnzPerRowColumn));
  return;
}

float host_cublasGEMV(bool TransA, float *A, float *x, float *y, int M, int N) {
  float *d_A = NULL, *d_x = NULL, *d_y = NULL;
  int size_A = M * N;
  int size_x = (TransA == false) ? N : M;
  int size_y = (TransA == false) ? M : N;
  CUDA_CHECK(cudaMalloc(
    (void**)&d_A,
    size_A * sizeof(d_A[0])
  ));
  CUDA_CHECK(cudaMalloc(
    (void**)&d_x,
    size_x  * sizeof(d_x[0])
  ));
  CUDA_CHECK(cudaMalloc(
    (void**)&d_y,
    size_y  * sizeof(d_y[0])
  ));

  CUDA_CHECK(cudaMemcpy(
    d_A,
    A,
    (size_t)(size_A  * sizeof(d_A[0])),
    cudaMemcpyHostToDevice
  ));
  CUDA_CHECK(cudaMemcpy(
    d_x,
    x,
    (size_t)(size_x  * sizeof(d_x[0])),
    cudaMemcpyHostToDevice
  ));
  CUDA_CHECK(cudaMemcpy(
    d_y,
    y,
    (size_t)(size_y  * sizeof(d_y[0])),
    cudaMemcpyHostToDevice
  ));
  float time = cublasGEMV(TransA, d_A, d_x, d_y, M, N);
  CUDA_CHECK(cudaMemcpy(
    y,
    d_y,
    (size_t)(size_y  * sizeof(d_y[0])),
    cudaMemcpyDeviceToHost
  ));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  return time;
}

float host_csrGEMV(bool TransA, float *A, float *x, float *y, int M, int N) {
  float *d_A = NULL, *d_x = NULL, *d_y = NULL;
  int size_A = M * N;
  int size_x = (TransA == false) ? N : M;
  int size_y = (TransA == false) ? M : N;
  CUDA_CHECK(cudaMalloc(
    (void**)&d_A,
    size_A * sizeof(d_A[0])
  ));
  CUDA_CHECK(cudaMalloc(
    (void**)&d_x,
    size_x  * sizeof(d_x[0])
  ));
  CUDA_CHECK(cudaMalloc(
    (void**)&d_y,
    size_y  * sizeof(d_y[0])
  ));

  CUDA_CHECK(cudaMemcpy(
    d_A,
    A,
    (size_t)(size_A  * sizeof(d_A[0])),
    cudaMemcpyHostToDevice
  ));
  CUDA_CHECK(cudaMemcpy(
    d_x,
    x,
    (size_t)(size_x  * sizeof(d_x[0])),
    cudaMemcpyHostToDevice
  ));
  CUDA_CHECK(cudaMemcpy(
    d_y,
    y,
    (size_t)(size_y  * sizeof(d_y[0])),
    cudaMemcpyHostToDevice
  ));
  float time = csrGEMV(TransA, d_A, d_x, d_y, M, N);
  CUDA_CHECK(cudaMemcpy(
    y,
    d_y,
    (size_t)(size_y  * sizeof(d_y[0])),
    cudaMemcpyDeviceToHost
  ));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  return time;
}


// GEMV with cuBLAS
// A size M*N, x size N*1, y size M*1
float cublasGEMV(bool TransA, float *A, float *x, float *y, int M, int N) {
  cublasHandle_t handle = NULL;
  CUBLAS_CHECK(cublasCreate(&handle));

  const float alpha = 1.0;
  const float beta = 0.0;

  // int lda = (TransA == false) ? M : N;
  cublasOperation_t cuTransA =
      (TransA == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  // int row = (TransA == false) ? M : N;
  // int col = (TransA == false) ? N : M;

  // struct timeval start, stop;
  // Timer timer;
  float total_time;

  cudaDeviceSynchronize();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // cudaEventRecord(start, 0);


  cudaEventRecord(start, 0);
  // timer.start();
  // wrong execution actually...
  CUBLAS_CHECK(cublasSgemv(handle,
    cuTransA,
    // row,
    // col,
    M,
    N,
    &alpha,
    A,
    // lda,
    M,
    x,
    1,
    &beta,
    y,
    1
  ));
  cudaDeviceSynchronize();
  // timer.end();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  CUDA_POST_KERNEL_CHECK;



  // total_time = timer.duration_ms();
  // LOG("cuBLAS Time: %f ms", total_time);
  cudaEventElapsedTime(&total_time, start, stop);
  LOG("cuBLAS GEMV total Time by cudaEvent: %f ms", total_time);

  return total_time;
}

// GEMV with cuSparse CSR format for matrix
float csrGEMV(bool TransA, float *A, float *x, float *y, int M, int N) {

  cusparseHandle_t handle = 0;
  CUSPARSE_CHECK(cusparseCreate(&handle));
  cusparseMatDescr_t descr = 0;
  CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  int nnz;
  float *csrVal;
  int *csrRowPtr;
  int *csrColIndex;

  // Note the different of row-major (host) and column-major (device)!
  int A_row_dev = M;
  int A_col_dev = N;
  convert_dense2csr(M, N, A, (void **)&csrVal, (void **)&csrRowPtr,
      (void **)&csrColIndex, &nnz);

  const float alpha = 1.0;
  const float beta = 0.0;

  cusparseOperation_t cuTransA =
      (TransA == false) ? CUSPARSE_OPERATION_NON_TRANSPOSE
                        : CUSPARSE_OPERATION_TRANSPOSE;


  float total_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaDeviceSynchronize();

  cudaEventRecord(start, 0);
  CUSPARSE_CHECK(cusparseScsrmv(handle,
    cuTransA,
    A_row_dev,
    A_col_dev,
    nnz,
    &alpha,
    descr,
    csrVal,
    csrRowPtr,
    csrColIndex,
    x,
    &beta,
    y));

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
  CUDA_POST_KERNEL_CHECK;
  cudaEventElapsedTime(&total_time, start, stop);
  LOG("cuSparse csrMV Time: %f ms", total_time);

  CUDA_CHECK(cudaFree(csrVal));
  CUDA_CHECK(cudaFree(csrRowPtr));
  CUDA_CHECK(cudaFree(csrColIndex));

  return total_time;
}
