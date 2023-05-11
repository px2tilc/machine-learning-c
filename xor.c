#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct {
  // Layer #1 neuron #1
  float or_w1;
  float or_w2;
  float or_b;

  // Layer #1 neuron #2
  float nand_w2;
  float nand_w1;
  float nand_b;

  // Layer #2 neuron #1
  float and_w2;
  float and_w1;
  float and_b;
} Xor;

float sigmoidf(float x)
{
  return 1.f / (1.f + expf(-x));
}

float forward(Xor m, float x1, float x2) 
{
  float a = sigmoidf(m.or_w1*x1 + m.or_w2*x2 + m.or_b);
  float b = sigmoidf(m.nand_w1*x1 + m.nand_w2*x2 + m.nand_b);
  return sigmoidf(m.and_w1*a + m.and_w2*b + m.and_b);
}

typedef float sample[3];

// XOR
sample xor_train[] = {
  {0,0,0},
  {0,1,1},
  {1,0,1},
  {1,1,0},
};

sample or_train[] = {
  {0,0,0},
  {0,1,1},
  {1,0,1},
  {1,1,1},
};

sample nor_train[] = {
  {0,0,1},
  {0,1,0},
  {1,0,0},
  {1,1,0},
};

sample and_train[] = {
  {0,0,0},
  {0,1,0},
  {1,0,0},
  {1,1,1},
};

sample nand_train[] = {
  {0,0,1},
  {0,1,1},
  {1,0,1},
  {1,1,0},
};

sample *train = nor_train;
#define train_count 4

float cost(Xor m)
{
  float result = 0.0f;
  for(size_t i = 0; i < train_count; ++i) {
    float x1 = train[i][0];
    float x2 = train[i][1];
    float y = forward(m, x1, x2);
    float d = y - train[i][2];
    result += d*d;
  }
  result /= train_count;
  return result;
}

float rand_float()
{
  return (float) rand() / (float) RAND_MAX;
}

Xor rand_xor(void)
{
  Xor m;
   // Layer #1 neuron #1
  m.or_w1 = rand_float();
  m.or_w2 = rand_float();
  m.or_b = rand_float();

  // Layer #1 neuron #2
  m.nand_w2 = rand_float();
  m.nand_w1 = rand_float();
  m.nand_b = rand_float();

  // Layer #2 neuron #1
  m.and_w2 = rand_float();
  m.and_w1 = rand_float();
  m.and_b = rand_float();
  return m;
}

void print_xor(Xor m)
{

  printf("or_w1 = %f\n", m.or_w1);
  printf("or_w2 = %f\n", m.or_w2);
  printf("or_b  = %f\n", m.or_b);
  
  printf("nand_w2 = %f\n", m.nand_w2);
  printf("nand_w1 = %f\n", m.nand_w1);
  printf("nand_b  = %f\n", m.nand_b);
  
  printf("and_w2 = %f\n", m.and_w2);
  printf("and_w1 = %f\n", m.and_w1);
  printf("and_b  = %f\n", m.and_b);

}

Xor apply_diff(Xor m, Xor g, float rate)
{
  m.or_w1 -= rate*g.or_w1;
  m.or_w2 -= rate*g.or_w2;
  m.or_b -= rate*g.or_b;
  m.nand_w2 -= rate*g.nand_w2;
  m.nand_w1 -= rate*g.nand_w1;
  m.nand_b -= rate*g.nand_b;
  m.and_w2 -= rate*g.and_w2;
  m.and_w1 -= rate*g.and_w1;
  m.and_b -= rate*g.and_b;
  return m;
}

Xor finite_diff(Xor m, float eps)
{ 
  Xor g;

  float c = cost(m);
  float saved; 

  saved = m.or_w1;
  m.or_w1 += eps;
  g.or_w1 = (cost(m) - c) / eps;
  m.or_w1 = saved;
  
  
  saved = m.or_w2;
  m.or_w2 += eps;
  g.or_w2 = (cost(m) - c) / eps;
  m.or_w2 = saved;

  saved = m.or_b;
  m.or_b += eps;
  g.or_b = (cost(m) - c) / eps;
  m.or_b = saved;

  saved = m.nand_w2;
  m.nand_w2 += eps;
  g.nand_w2 = (cost(m) - c) / eps;
  m.nand_w2 = saved;

  saved = m.nand_w1;
  m.nand_w1 += eps;
  g.nand_w1 = (cost(m) - c) / eps;
  m.nand_w1 = saved;
  
  saved = m.nand_b;
  m.nand_b += eps;
  g.nand_b = (cost(m) - c) / eps;
  m.nand_b = saved;
  
  saved = m.and_w2;
  m.and_w2 += eps;
  g.and_w2 = (cost(m) - c) / eps;
  m.and_w2 = saved;
  
  saved = m.and_w1;
  m.and_w1 += eps;
  g.and_w1 = (cost(m) - c) / eps;
  m.and_w1 = saved;
  
  saved = m.and_b;
  m.and_b += eps;
  g.and_b = (cost(m) - c) / eps;
  m.and_b = saved;

  return g;
}

int normalizef(float x) {
  if (x > 0.5f) {
    return 1;    
  }
  return 0;
}

int main()
{
  srand(time(0));
  
  Xor m = rand_xor(); 
  
  
  float eps = 0.1;
  float rate = 0.1;

  
  for(size_t i= 0; i< 100*1000; ++i) {
    Xor diff = finite_diff(m, eps);
    m = apply_diff(m, diff, rate);
    printf("cost: %f\n", cost(m));
  }
  printf("cost: %f\n", cost(m));

  printf("---------------------------\n");
  for(size_t i = 0; i < 2; ++i) {
    for(size_t j = 0; j < 2; ++j) {

      int result = normalizef(forward(m, i, j));
      
      printf("%zu ^ %zu => %i\n", i, j, result);
    }
  }

  // printf("---------------------------\n");
  // printf("\"OR\" neuron:\n");
  // for(size_t i = 0; i < 2; ++i) {
  //   for(size_t j = 0; j < 2; ++j) {
  //     float orf = sigmoidf(m.or_w1 * i + m.or_w2 * j  + m.or_b);
  //     printf("%zu | %zu => %i\n", i, j, normalizef(orf));
  //   }
  // }
  // printf("---------------------------\n");
  // printf("\"NAND\" neuron:\n");
  // for(size_t i = 0; i < 2; ++i) {
  //   for(size_t j = 0; j < 2; ++j) {
  //     float orf = sigmoidf(m.nand_w1 * i + m.nand_w2 * j  + m.nand_b);
  //     printf("~(%zu & %zu) => %i\n", i, j, normalizef(orf));
  //   }
  // }
  // printf("---------------------------\n");
  // printf("\"AND\" neuron:\n");
  // for(size_t i = 0; i < 2; ++i) {
  //   for(size_t j = 0; j < 2; ++j) {
  //     float orf = sigmoidf(m.and_w1 * i + m.and_w2 * j  + m.and_b);
  //     printf("%zu & %zu => %i\n", i, j, normalizef(orf));
  //   }
  // }
  return 0;
}