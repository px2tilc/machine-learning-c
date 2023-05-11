#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

float train[][2] = {
  {0, 0},
  {1, 2},
  {2, 4},
  {3, 6},
  {4, 8},
};
#define train_count (sizeof(train)/sizeof(train[0]))

float rand_float() 
{
  return (float) rand() / (float) RAND_MAX;
}

float cost(float w)
{
  float result = 0.0f;
  for (size_t i = 0; i < train_count; ++i) {
    float x = train[i][0];
    float y = x*w;
    float d = y - train[i][1];
    result += d*d;
  }
  result /= train_count;  
  return result;
}

float gcost(float w)
{
  float result = 0.0f;
  size_t n = train_count;
  for(size_t i = 0; i < n; ++i)
  {
    float x = train[i][0];
    float y = train[i][1];
    result += 2 * (x*w - y) * x;
  }
  result /= n;
  return result;
}

// y = x*w;
int main()
{
  srand(time(0));
  float w = rand_float()*10.0f;
  // float b = rand_float()*5.0f;

  // float eps = 0.001;
  float rate = 0.1;
  
  printf("cost = %f, w = %f\n", gcost(w), w);
  for(size_t i = 0; i < 8; ++i){
#if 1
    float eps = 0.001;
    float c = cost(w);
    float dw = (cost(w + eps) - c) / eps;
#else 
    float dw = gcost(w);
#endif
    w -= rate*dw;
    printf("cost = %f, w = %f\n", gcost(w), w);
  }
  printf("-----------------------\n");
  printf("cost = %f\n", gcost(w));
  printf("w = %f\n", w);

  return 0;
}
