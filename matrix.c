#include "matrix.h"

Matrix::Matrix()
{
  row_ = 0; col_ = 0; size_ = 0; nnz_ = 0;
  rowIdx_ = NULL; colIdx_ = NULL; rowPtr_ = NULL; colPtr_ = NULL;
  mat_ = NULL; val_ = NULL;
  inited_ =false; isDENSE_ = false; isCSR_ = false; isCOO_ = false; isCSC_ = false;
}

Matrix::~Matrix()
{
  if (rowIdx_ != NULL) delete [] rowIdx_;
  if (colIdx_ != NULL) delete [] colIdx_;
  if (rowPtr_ != NULL) delete [] rowPtr_;
  if (colPtr_ != NULL) delete [] colPtr_;
  if (mat_ != NULL) delete [] mat_;
  if (val_ != NULL) delete [] val_;
}

void Matrix::input_from_fp(int format, FILE *& fp)
{
  switch (format) {
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
      printf("format error: format not defined\n");
      exit(1);
    }
  }
  inited_ = true;
}

void Matrix::output_to_fp(int format, FILE *& fp)
{
  if (!inited_) {
    printf("error: outputing empty matrix %s\n", name_.c_str());
    exit(1);
  }
  switch (format) {
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
      printf("format error: format not defined\n");
      exit(1);
    }
  }
  
}

void Matrix::convert_to_dense()
{
  if (isDENSE_) return;
  if (!inited_) {
    printf("error: converting empty matrix %s to dense format\n", name_.c_str());
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
        colIdx_[nnz_cnt] = mat_[cnt];
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
