#ifndef NL_H_
#define NL_H_

#include <stddef.h>
#include <stdio.h>

#ifndef NL_MALLOC
#include <stdlib.h>
#define NL_MALLOC(a) (float*)malloc(a);
#endif //NL_MALLOC

#ifndef NL_ASSERT
#include <assert.h>
#define NL_ASSERT(a) assert(a);
#endif //NL_ASSERT

// float d[] = {0,1,0,1}
// Mat m = { .rows=2, .cols=4, .stride=3, .es=&d[0] };
// Mat m = { .rows=2, .cols=4, .stride=3, .es=&d[3] };

typedef struct {
  size_t rows;
  size_t cols;
  float *es;
} Mat;

#define MAT_AT(m, i, j) m.es[(i)*(m).cols + (j)]

float rand_float(void);

Mat  mat_alloc(size_t rows, size_t cols);
void mat_rand(Mat m, float low, float high);
void mat_dot(Mat dest, Mat a, Mat b);
void mat_sum(Mat dest, Mat a);
void mat_print(Mat m);

#endif // NL_H_

#ifdef NL_IMPLEMENTATION

float rand_float()
{
  return (float)rand() / (float)RAND_MAX;
}

Mat  mat_alloc(size_t rows, size_t cols)
{
   Mat m;
   m.rows = rows,
   m.cols = cols;
   m.es = NL_MALLOC(sizeof(*m.es)*rows*cols);
   NL_ASSERT(m.es != NULL);
   return m;
}

void mat_dot(Mat dest, Mat a, Mat b)
{
  (void) dest;
  (void) a;
  (void) b;
}

void mat_sum(Mat dest, Mat a)
{
  (void) dest;
  (void) a;
}

void mat_print(Mat m)
{
  printf("\n");
  for(size_t i = 0; i < m.rows; ++i) {
    for(size_t j = 0; j < m.cols; ++j) {
      printf("%f ", MAT_AT(m, i, j));
    }
    printf("\n");
  }
  printf("\n");
}

void mat_rand(Mat m, float low, float high) 
{
  for(size_t i = 0; i < m.rows; ++i) {
    for(size_t j = 0; j < m.cols; ++j) {
      MAT_AT(m, i, j) = rand_float()*(high - low) + low;
    }
  }
}

#endif //NL_H_