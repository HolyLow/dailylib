#ifndef __MATRIX_COUPLE_H__
 #define __MATRIX_COUPLE_H__

#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>


class MatrixCouple {
public:
  MatrixCouple() {
    a.set_name("left_mat");
    b.set_name("right_mat");
    c.set_name("result_mat");
  }
  void input_from_file(char *filename, int format_a, int format_b);
  void output_to_file(char *filename, int format_a, int format_b);
private:
  Matrix a;
  Matrix b;
  Matrix c;
};
#endif
