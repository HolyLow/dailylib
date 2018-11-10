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
#define COC     4   // Compressed Ordered Columns

#define NULL_IDX  0

inline std::string format(int f) {
  switch (f) {
    case DENSE: return "DENSE";
    case COO:   return "COO";
    case CSR:   return "CSR";
    case COC:   return "COC";
    default:    return "NON_DEF_FORMAT";
  }
}

struct Dense {
  float* val;
  int row;
  int col;
  Dense() { val = NULL; row = 0; col = 0; }
  void print();
  void print_param();
};

struct Csr {
  float* val;
  int* rowPtr;
  int* colIdx;
  int row;
  int col;
  int nnz;
  Csr() { val = NULL; rowPtr = NULL; colIdx = NULL; row = 0; col = 0; nnz = 0; }
  void print();
  void print_param();
};

struct Coo {
  float* val;
  int* rowIdx;
  int* colIdx;
  int row;
  int col;
  int nnz;
  Coo() { val = NULL; rowIdx = NULL; colIdx = NULL; row = 0; col = 0; nnz = 0; }
  void print();
  void print_param();
};

struct Coc {
  float* val;
  int* colIdx;
  int row;
  int col;
  int nnz;
  int comp_col;
  Coc() { val = NULL; colIdx = NULL; row = 0; col = 0; nnz = 0; comp_col = 0; }
  void print();
  void print_param();
};


class Matrix {
public:
  Matrix();
  ~Matrix();
  void init(int m, int n, float sp = 0.0);
  void row_balance_init(int m, int n, float sp = 0.0);

  void set_name(std::string p) { name_ = p; }
  int row() { return row_; }
  int col() { return col_; }
  void input_from_file(int f, char* filename);
  void output_to_file(int f, char* filename);
  void convert_to_dense();
  void convert_to_csr();
  void convert_to_coo();
  void convert_to_coc();
  Dense get_dense();
  Csr get_csr();
  Coo get_coo();
  Coc get_coc();


  bool is_format(int f) {
    switch(f) {
      case DENSE: return isDENSE_;
      case CSR:   return isCSR_;
      case COO:   return isCOO_;
      // case CSC:   return isCSC_;
      case COC:   return isCOC_;
      default:    return false;
    }
  }

private:
  void dense_to_csr();
  void csr_to_coo();
  void coo_to_csr();
  void coo_to_dense();
  void dense_to_coc();

  void reset();



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

  int coc_col_;
  int* coc_colIdx_;
  float* coc_val_;
  bool isCOC_;

};
#endif
