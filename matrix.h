#ifndef __MATRIX_H__
 #define __MATRIX_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#define DENSE   0
#define COO     1
#define CSR     2
// #define CSC     3

class Matrix {
public:
  Matrix();
  ~Matrix();

  void set_name(std::string p) { name_ = p; }
  int row() { return row_; }
  int col() { return col_; }
  int nnz() { return nnz_; }
  int* rowIdx() { return rowIdx_; }
  int* colIdx() { return colIdx_; }
  int* rowPtr() { return rowPtr_; }
  int* colPtr() { return colPtr_; }
  float* mat() { return mat_; }
  float* val() { return val_; }
  void input_from_fp(int format, FILE *& fp);
  void output_to_fp(int format, FILE *& fp);
  void convert_to_dense();
  void convert_to_csr();
  void convert_to_coo();

  bool is_format(int format) {
    switch(format) {
      case DENSE: return isDENSE_;
      case CSR:   return isCSR_;
      case COO:   return isCOO_;
      // case CSC:   return isCSC_;
      default:    return false;
    }
  }



private:

  void dense_to_csr();
  void csr_to_coo();
  void coo_to_csr();
  void coo_to_dense();



  std::string name_;
  int row_;
  int col_;
  int size_;
  int nnz_;
  int *rowIdx_;
  int *colIdx_;
  int *rowPtr_;
  int *colPtr_;
  float *mat_;
  float *val_;

  bool inited_;
  bool isDENSE_;
  bool isCSR_;
  bool isCOO_;
  bool isCSC_;
};
#endif
