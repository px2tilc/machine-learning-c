#define NL_IMPLEMENTATION
#include "nl.h"
#include <time.h>

// Training data
float td_xor[] = {
  0, 0, 0,
  0, 1, 1,
  1, 0, 1,
  1, 1, 0,
};
float td_or[] = {
  0, 0, 0,
  0, 1, 1,
  1, 0, 1,
  1, 1, 1,
};
float td_and[] = {
  0, 0, 0,
  0, 1, 0,
  1, 0, 0,
  1, 1, 1,
};
float td_nand[] = {
  0, 0, 1,
  0, 1, 1,
  1, 0, 1,
  1, 1, 0,
};

int main()
{
  srand(time(0));
  
  float *td = td_xor;

  float eps = 0.1;
  float rate = 0.1;

  size_t stride = 3;
  size_t n = 4;
  
  Mat ti = {
    .rows = n,
    .cols = 2,
    .stride = stride,
    .es = td,
  }; 

  Mat to = {
    .rows = n,
    .cols = 1,
    .stride = stride,
    .es = td + 2,
  };
  
  // This array 'arch' describes the architecture of the neural network
  // {input/s, ...layer neuron/s..., output/s}
  size_t arch[] = {2, 10, 1};
  NN nn = nn_alloc(arch, ARRAY_LEN(arch));
  NN gn = nn_alloc(arch, ARRAY_LEN(arch));
  nn_rand(nn, 0, 1);
   
  for(size_t i = 0; i < 20 * 1000; ++i) {
    nn_finite_diff(nn, gn, eps, ti, to);
    nn_learn(nn, gn, rate);
    printf("cost = %f\n", nn_cost(nn, ti, to));
  }

  for(size_t i = 0; i < 2; ++i) {
    for(size_t j = 0; j < 2; ++j) {
      MAT_AT(NN_INPUT(nn), 0, 0) = i;
      MAT_AT(NN_INPUT(nn), 0, 1) = j;
      nn_forward(nn);
      float result = MAT_AT(NN_OUTPUT(nn), 0, 0);
      if (result < 0.5) {
        result = 0;
      } else {
        result = 1;
      }
      printf("%zu ^ %zu = %f\n", i, j, result);
    } 
  }
  
  return 0;
}