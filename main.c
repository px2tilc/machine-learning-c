#include <stdio.h>
#define NEURALIB_IMPLEMENTATION
#include "neuralib.h"

int main()
{
  Mat m = mat_alloc(4, 4);

  mat_print(m);

  return 0;
}