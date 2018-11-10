#ifndef COMMON_CUDA_H_
 #define COMMON_CUDA_H_

#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>

#define CUDA_CHECK(status_) do {                                              \
    cudaError_t status = (status_);                                           \
    if (status != cudaSuccess) {                                              \
      fprintf(stderr, "<%s : %d> %s: check cuda success error, %s\n",         \
          __FILE__, __LINE__, __FUNCTION__, cudaGetErrorString(status));      \
    }                                                                         \
} while(0)

// CUDA: check for error after kernel execution and exit loudly if there is one.
// #define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaGetLastError())

#define CUDNN_CHECK(status_) do {                                             \
    cudnnStatus_t status = (status_);                                         \
    if (status != CUDNN_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "<%s : %d> %s: check cudnn success error, %s\n",        \
          __FILE__, __LINE__, __FUNCTION__, cudnnGetErrorString(status));     \
    }                                                                         \
} while(0)

// as there is no error-string api in cublas, a private one is implemented
inline const char* cublasMyGetErrorString(cublasStatus_t status)
{
  switch(status)
  {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  return "unknown error";
}

#define CUBLAS_CHECK(status_) do {                                            \
    cublasStatus_t status = (status_);                                        \
    if (status != CUBLAS_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "<%s : %d> %s: check cublas success error, %s\n",       \
          __FILE__, __LINE__, __FUNCTION__, cublasMyGetErrorString(status));  \
    }                                                                         \
} while(0)

inline const char* cusparseMyGetErrorString(cusparseStatus_t status)
{
  switch (status) {
    case CUSPARSE_STATUS_SUCCESS:
      return "CUSPARSE_STATUS_SUCCESS";
    case CUSPARSE_STATUS_NOT_INITIALIZED:
      return "CUSPARSE_STATUS_NOT_INITIALIZED";
    case CUSPARSE_STATUS_ALLOC_FAILED:
      return "CUSPARSE_STATUS_ALLOC_FAILED";
    case CUSPARSE_STATUS_INVALID_VALUE:
      return "CUSPARSE_STATUS_INVALID_VALUE";
    case CUSPARSE_STATUS_ARCH_MISMATCH:
      return "CUSPARSE_STATUS_ARCH_MISMATCH";
    case CUSPARSE_STATUS_MAPPING_ERROR:
      return "CUSPARSE_STATUS_MAPPING_ERROR";
    case CUSPARSE_STATUS_EXECUTION_FAILED:
      return "CUSPARSE_STATUS_EXECUTION_FAILED";
    case CUSPARSE_STATUS_INTERNAL_ERROR:
      return "CUSPARSE_STATUS_INTERNAL_ERROR";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  }
  return "unknown error";
}

#define CUSPARSE_CHECK(status_) do {                                          \
    cusparseStatus_t status = (status_);                                      \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                  \
      fprintf(stderr, "<%s : %d> %s: check cusparse success error, %s\n",     \
          __FILE__, __LINE__, __FUNCTION__, cusparseMyGetErrorString(status));\
    }                                                                         \
} while(0)

#endif
