#include "matrix.h"

#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void Dense::print()
{
  if (val != NULL) {
    printf("row = %d, col = %d\nin dense format:\n", row, col);
    for (int i = 0, cnt = 0; i < row; ++i) {
      for (int j = 0; j < col; ++j, ++cnt) {
        printf("%.2f\t", val[cnt]);
      }
      printf("\n");
    }
  }
  else printf("dense not inited\n");
}

void Dense::print_param()
{
  if (val != NULL) {
    printf("in dense format, row = %d, col = %d\n", row, col);
  }
  else printf("dense not inited\n");
}

void Csr::print()
{
  if (val != NULL) {
    printf("row = %d, col = %d, nnz = %d\nin csr format:\n", row, col, nnz);
    printf("val\t");
    for (int i = 0; i < nnz; ++i) printf("%.2f\t", val[i]);
    printf("\n");
    printf("colIdx\t");
    for (int i = 0; i < nnz; ++i) printf("%d\t", colIdx[i]);
    printf("\n");
    printf("rowPtr\t");
    for (int i = 0; i <= row; ++i) printf("%d\t", rowPtr[i]);
    printf("\n");
  }
  else printf("csr not inited\n");
}

void Csr::print_param()
{
  if (val != NULL) {
    printf("in csr format, row = %d, col = %d, nnz = %d\n", row, col, nnz);
  }
  else printf("csr not inited\n");
}

void Coo::print()
{
  if (val != NULL) {
    printf("row = %d, col = %d, nnz = %d\nin coo format:\n", row, col, nnz);
    printf("val\t");
    for (int i = 0; i < nnz; ++i) printf("%.2f\t", val[i]);
    printf("\n");
    printf("rowIdx\t");
    for (int i = 0; i < nnz; ++i) printf("%d\t", rowIdx[i]);
    printf("\n");
    printf("colIdx\t");
    for (int i = 0; i < nnz; ++i) printf("%d\t", colIdx[i]);
    printf("\n");
  }
  else printf("coo not inited\n");
}

void Coo::print_param()
{
  if (val != NULL) {
    printf("in coo format, row = %d, col = %d, nnz = %d\n", row, col, nnz);
  }
  else printf("coo not inited\n");
}

void Coc::print()  // assume COC is column major
{
  if (val != NULL) {
    printf("row = %d, col = %d, nnz = %d, comp_col = %d\nin coc format:\n",
        row, col, nnz, comp_col);
    printf("val\n");
    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < comp_col; ++j) {
        printf("%.2f\t", val[j * row + i]);
      }
      printf("\n");
    }
    printf("colIdx\n");
    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < comp_col; ++j) {
        printf("%d\t", colIdx[j * row + i]);
      }
      printf("\n");
    }
  }
  else printf("coc not inited\n");
}

void Coc::print_param()  // assume COC is column major
{
  if (val != NULL) {
    printf("in coc format, "
        "row = %d, col = %d, nnz = %d, comp_col = %d, efficiency = %.2f\%\n",
        row, col, nnz, comp_col, (float)nnz / (row * comp_col) * 100);
  }
  else printf("coc not inited\n");
}

void rand_pick_k_nums_from_n(int* array, int k, int n, int rseed = 0)
{
  int* flag = new int[n];
  for(int i = 0; i < n; ++i) {
    flag[i] = i;
  }
  srand((unsigned)time(0) + rseed);
  for (int i = 0; i < k; ++i) {
    int r = (rand() % (n - i)) + i;
    int tmp = flag[r];
    flag[r] = flag[i];
    flag[i] = tmp;
  }
  for (int i = 0; i < k; ++i) {
    array[i] = flag[i];
  }
  delete [] flag;
}

void generate_sparse_matrix(float* mat, int m, int n, float sparsity, int rseed = 0)
{
  int size = m * n;
  // float* mat = new float[size];
  for (int i = 0; i < size; ++i) { mat[i] = 0.; }
  int cnt = (1 - sparsity) * size;
  if (cnt == 0 && sparsity != 1.0) cnt = 1;
  int* idx = new int[cnt];
  rand_pick_k_nums_from_n(idx, cnt, size, rseed);
  // sleep(0.01); // to change the time for rand seed
  srand((unsigned)time(0) + rseed);
  for (int i = 0; i < cnt; ++i) {
    mat[idx[i]] = rand() % 100 + 1;
  }
  delete [] idx;
}

Matrix::Matrix()
{
  name_ = "matrix";
  row_ = 0; col_ = 0; size_ = 0; nnz_ = 0;
  rowIdx_ = NULL; colIdx_ = NULL; rowPtr_ = NULL; colPtr_ = NULL;
  mat_ = NULL; val_ = NULL;
  inited_ =false; isDENSE_ = false; isCSR_ = false; isCOO_ = false;
  isCSC_ = false;

  coc_col_ = 0; coc_colIdx_ = NULL; coc_val_ = NULL; isCOC_ = false;
}

Matrix::~Matrix()
{
  if (rowIdx_ != NULL) delete [] rowIdx_;
  if (colIdx_ != NULL) delete [] colIdx_;
  if (rowPtr_ != NULL) delete [] rowPtr_;
  if (colPtr_ != NULL) delete [] colPtr_;
  if (mat_ != NULL) delete [] mat_;
  if (val_ != NULL) delete [] val_;

  if (coc_val_ != NULL) delete [] coc_val_;
  if (coc_colIdx_ != NULL) delete [] coc_colIdx_;
}

void Matrix::init(int m, int n, float sp)
{
  if (inited_) reset();
  mat_ = new float[m * n];
  generate_sparse_matrix(mat_, m, n, sp);
  inited_ = true;
  size_ = m * n;
  row_ = m; col_ = n;
  isDENSE_ = true;
}

void Matrix::row_balance_init(int m, int n, float sp)
{
  if (inited_) reset();
  mat_ = new float[m * n];
  for (int i = 0; i < m; ++i) {
    generate_sparse_matrix(&mat_[i * n], 1, n, sp, i);
  }
  // generate_sparse_matrix(mat_, m, n, sp);
  inited_ = true;
  size_ = m * n;
  row_ = m; col_ = n;
  isDENSE_ = true;
}

void Matrix::input_from_file(int f, char* filename)
{
  if (inited_) reset();
  FILE* fp = fopen(filename, "rb");
  if (fp == NULL) {
    printf("open %s failed\n", filename);
    return;
  }
  switch (f) {
    case DENSE: {
      isDENSE_ = true;
      fread((void*)&row_, sizeof(int), 1, fp);
      fread((void*)&col_, sizeof(int), 1, fp);
      size_ = row_ * col_;
      mat_ = new float[size_];
      fread((void*)mat_, sizeof(float), size_, fp);
      break;
    }
    case CSR: {
      isCSR_ = true;
      fread((void*)&row_, sizeof(int), 1, fp);
      fread((void*)&col_, sizeof(int), 1, fp);
      size_ = row_ * col_;
      fread((void*)&nnz_, sizeof(int), 1, fp);
      val_ = new float[nnz_];
      rowPtr_ = new int[row_ + 1];
      rowPtr_[0] = 0;
      colIdx_ = new int[nnz_];
      fread((void*)val_, sizeof(val_[0]), nnz_, fp);
      fread((void*)rowPtr_, sizeof(rowPtr_[0]), row_ + 1, fp);
      fread((void*)colIdx_, sizeof(colIdx_[0]), nnz_, fp);
      break;
    }
    case COO: {
      isCOO_ = true;
      fread((void*)&row_, sizeof(int), 1, fp);
      fread((void*)&col_, sizeof(int), 1, fp);
      size_ = row_ * col_;
      fread((void*)&nnz_, sizeof(int), 1, fp);
      val_ = new float[nnz_];
      rowIdx_ = new int[nnz_];
      colIdx_ = new int[nnz_];
      fread((void*)val_, sizeof(val_[0]), nnz_, fp);
      fread((void*)rowIdx_, sizeof(rowIdx_[0]), nnz_, fp);
      fread((void*)colIdx_, sizeof(colIdx_[0]), nnz_, fp);
      break;
    }

    default: {
      printf("input format %s(%d) error: format not defined or supported\n",
          format(f).c_str(), f);
      fclose(fp);
      exit(1);
    }
  }
  fclose(fp);
  inited_ = true;
}

void Matrix::output_to_file(int f, char* filename)
{
  if (!inited_) {
    printf("error: outputing empty matrix %s\n", name_.c_str());
    exit(1);
  }
  FILE* fp = fopen(filename, "wb");
  if (fp == NULL) {
    printf("open %s failed\n", filename);
    return;
  }
  switch (f) {
    case DENSE: {
      convert_to_dense();
      fwrite((void*)&row_, sizeof(int), 1, fp);
      fwrite((void*)&col_, sizeof(int), 1, fp);
      fwrite((void*)mat_, sizeof(float), size_, fp);
      break;
    }
    case CSR: {
      convert_to_csr();
      fwrite((void*)&row_, sizeof(int), 1, fp);
      fwrite((void*)&col_, sizeof(int), 1, fp);
      fwrite((void*)&nnz_, sizeof(int), 1, fp);
      fwrite((void*)val_, sizeof(val_[0]), nnz_, fp);
      fwrite((void*)rowPtr_, sizeof(rowPtr_[0]), row_ + 1, fp);
      fwrite((void*)colIdx_, sizeof(colIdx_[0]), nnz_, fp);
      break;
    }
    case COO: {
      convert_to_coo();
      fwrite((void*)&row_, sizeof(int), 1, fp);
      fwrite((void*)&col_, sizeof(int), 1, fp);
      fwrite((void*)&nnz_, sizeof(int), 1, fp);
      fwrite((void*)val_, sizeof(val_[0]), nnz_, fp);
      fwrite((void*)rowIdx_, sizeof(rowIdx_[0]), nnz_, fp);
      fwrite((void*)colIdx_, sizeof(colIdx_[0]), nnz_, fp);
      break;
    }

    default: {
      printf("output format %s(%d) error: format not defined or supported\n",
          format(f).c_str(), f);
      fclose(fp);
      exit(1);
    }
  }
  fclose(fp);

}

void Matrix::convert_to_dense()
{
  if (isDENSE_) return;
  if (!inited_) {
    printf("error: converting empty matrix %s to dense format\n",
        name_.c_str());
    exit(1);
  }
  convert_to_coo();
  coo_to_dense();
}

void Matrix::convert_to_csr()
{
  if (isCSR_) return;
  if (!inited_) {
    printf("error: converting empty matrix %s to csr format\n", name_.c_str());
    exit(1);
  }
  if (isCOO_) {
    coo_to_csr();
  }
  else {
    convert_to_dense();
    dense_to_csr();
  }
}

void Matrix::convert_to_coo()
{
  if (isCOO_) return;
  if (!inited_) {
    printf("error: converting empty matrix %s to coo format\n", name_.c_str());
    exit(1);
  }
  convert_to_csr();
  csr_to_coo();
}

void Matrix::convert_to_coc()
{
  if (isCOC_) return;
  if (!inited_) {
    printf("error: converting empty matrix %s to coc format\n", name_.c_str());
    exit(1);
  }
  convert_to_dense();
  dense_to_coc();
}

Dense Matrix::get_dense()
{
  Dense mat;
  convert_to_dense();
  mat.val = mat_;
  mat.row = row_;
  mat.col = col_;
  return mat;
}

Csr Matrix::get_csr()
{
  Csr mat;
  convert_to_csr();
  mat.val = val_;
  mat.rowPtr = rowPtr_;
  mat.colIdx = colIdx_;
  mat.row = row_;
  mat.col = col_;
  mat.nnz = nnz_;
  return mat;
}

Coo Matrix::get_coo()
{
  Coo mat;
  convert_to_coo();
  mat.val = val_;
  mat.rowIdx = rowIdx_;
  mat.colIdx = colIdx_;
  mat.row = row_;
  mat.col = col_;
  mat.nnz = nnz_;
  return mat;
}

Coc Matrix::get_coc()
{
  Coc mat;
  convert_to_coc();
  mat.val = coc_val_;
  mat.colIdx = coc_colIdx_;
  mat.row = row_;
  mat.col = col_;
  mat.nnz = nnz_;
  mat.comp_col = coc_col_;
  return mat;
}

void Matrix::dense_to_csr()
{
  nnz_ = 0;
  for (int i = 0, cnt = 0; i < row_; ++i) {
    for (int j = 0; j < col_; ++j, ++cnt) {
      if (mat_[cnt] != 0.0) {
        ++nnz_;
      }
    }
  }
  val_ = new float[nnz_];
  rowPtr_ = new int[row_ + 1];
  rowPtr_[0] = 0;
  colIdx_ = new int[nnz_];
  for (int i = 0, cnt = 0, nnz_cnt = 0; i < row_; ++i) {
    for (int j = 0; j < col_; ++j, ++cnt) {
      if (mat_[cnt] != 0.0) {
        val_[nnz_cnt] = mat_[cnt];
        colIdx_[nnz_cnt] = j;
        ++nnz_cnt;
      }
    }
    rowPtr_[i + 1] = nnz_cnt;
  }
  isCSR_ = true;
}

void Matrix::csr_to_coo()
{
  rowIdx_ = new int[nnz_];
  for (int i = 0; i < row_; ++i) {
    for (int j = rowPtr_[i]; j < rowPtr_[i + 1]; ++j) {
      rowIdx_[j] = i;
    }
  }
  isCOO_ = true;
}

void Matrix::coo_to_csr()
{
  rowPtr_ = new int[row_ + 1];
  rowPtr_[0] = 0;
  for (int i = 0; i < nnz_; ++i) {
    ++rowPtr_[rowIdx_[i] + 1];
  }
  for (int i = 1; i <= row_; ++i) {
    rowPtr_[i] += rowPtr_[i - 1];
  }
  isCSR_ = true;
}

void Matrix::coo_to_dense()
{
  mat_ = new float[size_];
  for (int i = 0; i < size_; ++i) {
    mat_[i] = 0.0;
  }
  for (int i = 0; i < nnz_; ++i) {
    mat_[ rowIdx_[i] * col_ + colIdx_[i] ] = val_[i];
  }
  isDENSE_ = true;
}

void Matrix::dense_to_coc()
{
  nnz_ = 0;
  coc_col_ = 0;
  for (int i = 0, cnt = 0; i < row_; ++i) {
    int row_cnt = 0;
    for (int j = 0; j < col_; ++j, ++cnt) {
      if (mat_[cnt] != 0.0) {
        ++nnz_;
        ++row_cnt;
      }
    }
    if (row_cnt > coc_col_) coc_col_ = row_cnt;
  }
  int coc_size = coc_col_ * row_;
  coc_val_ = new float[coc_size];
  coc_colIdx_ = new int[coc_size];
  for (int i = 0, cnt = 0; i < row_; ++i) {
    int row_cnt = 0;
    for (int j = 0; j < col_; ++j, ++cnt) {
      if (mat_[cnt] != 0.0) {
        coc_val_[row_cnt * row_ + i] = mat_[cnt];
        coc_colIdx_[row_cnt * row_ + i] = j;
        ++row_cnt;
      }
    }
    while (row_cnt < coc_col_) {
      coc_val_[row_cnt * row_ + i] = 0.0;
      coc_colIdx_[row_cnt * row_ + i] = NULL_IDX;
      ++row_cnt;
    }
  }
  isCOC_ = true;
}

void Matrix::reset()
{
  if (rowIdx_ != NULL) delete [] rowIdx_;
  if (colIdx_ != NULL) delete [] colIdx_;
  if (rowPtr_ != NULL) delete [] rowPtr_;
  if (colPtr_ != NULL) delete [] colPtr_;
  if (mat_ != NULL) delete [] mat_;
  if (val_ != NULL) delete [] val_;

  if (coc_val_ != NULL) delete [] coc_val_;
  if (coc_colIdx_ != NULL) delete [] coc_colIdx_;

  row_ = 0; col_ = 0; size_ = 0; nnz_ = 0;
  rowIdx_ = NULL; colIdx_ = NULL; rowPtr_ = NULL; colPtr_ = NULL;
  mat_ = NULL; val_ = NULL;
  inited_ =false; isDENSE_ = false; isCSR_ = false; isCOO_ = false;
  isCSC_ = false;

  coc_col_ = 0; coc_colIdx_ = 0; coc_val_ = NULL; isCOC_ = false;
}
