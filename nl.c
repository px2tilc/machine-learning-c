#define NL_IMPLEMENTATION
#include "nl.h"
#include <time.h>

int main()
{
  srand(time(0));

  Mat a = mat_alloc(1,2);
  mat_rand(a, 5, 10);
  
  float id_data[4] = {
    1, 0,
    0, 1,
  };
  
  Mat b = {
    .rows = 2,
    .cols = 2,
    .es = id_data
  };

  Mat dst = mat_alloc(1, 2);
  
  mat_print(a);
  printf("-------------------------\n");
  mat_dot(dst, a, b);

  mat_print(dst);
  return 0;
}