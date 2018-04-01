#include "rand_gen.h"
#include <string.h>

int* rand_pick_k_nums_from_n(int k, int n)
{
  int* flag = new int[n];
  int* array = new int[k];
  for(int i = 0; i < n; ++i){
    flag[i] = i;
  }
  srand((unsigned)time(0));
  for(int i = 0; i < k; ++i){
    int r = (rand() % (n - i)) + i;
    int tmp = flag[r];
    flag[r] = flag[i];
    flag[i] = tmp;
  }
  for(int i = 0; i < k; ++i){
    array[i] = flag[i];
  }
  delete [] flag;
  return array;
}

int* generate_sparse_matrix(int m, int n, float sparsity)
{
  int* mat = new int[m*n];
  memset(mat, 0, sizeof(int)*m*n);
  int cnt = (1-sparsity) * m * n;
  int* idx = rand_pick_k_nums_from_n(cnt, m*n);
  srand((unsigned)time(0));
  for(int i = 0; i < cnt; ++i){
    mat[idx[i]] = rand() + 1;
  }
  delete [] idx;
  return mat;
}
