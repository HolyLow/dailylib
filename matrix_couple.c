#include "matrix_couple.h"

void MatrixCouple::input_from_file(char *filename,
      int format_a, int format_b)
{
  FILE *fp;
  fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("error: input file %s does not exist\n", filename);
    exit(1);
  }
  a.input_from_fp(format_a, fp);
  b.input_from_fp(format_b, fp);
  fclose(fp);
  if (a.col() != b.row()) {
    printf("error: left matrix %d*%d does not match right matrix %d*%d\n",
            a.row(), a.col(), b.row(), b.col());
    exit(1);
  }
}

void MatrixCouple::output_to_file(char *filename, int format_a, int format_b)
{
  FILE *fp;
  fp = fopen(filename, "w");
  if (fp == NULL) {
    printf("error: open output file %s failed\n", filename);
    exit(1);
  }
  a.output_to_fp(format_a, fp);
  b.output_to_fp(format_b, fp);
  flose(fp);
}
