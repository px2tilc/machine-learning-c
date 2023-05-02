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
void mat_fill(Mat m, float x);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
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

void mat_dot(Mat dst, Mat a, Mat b)
{
  NL_ASSERT(a.cols == b.rows);
  size_t n = a.cols;
  NL_ASSERT(dst.rows == a.rows);
  NL_ASSERT(dst.cols == b.cols);
  

  for(size_t i = 0; i < dst.rows; ++i) {
    for(size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) = 0;
      for(size_t k = 0; k < n; ++k) {
        MAT_AT(dst, i, j) += MAT_AT(a, i, k)*MAT_AT(b, k, j);
      }      
    }
  }
}

void mat_sum(Mat dst, Mat a)
{
  NL_ASSERT(dst.rows == a.rows);
  NL_ASSERT(dst.cols == a.cols);
  for(size_t i = 0; i < dst.rows; ++i) {
    for(size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) += MAT_AT(a, i, j);
    }
  }
}

void mat_print(Mat m)
{
  for(size_t i = 0; i < m.rows; ++i) {
    for(size_t j = 0; j < m.cols; ++j) {
      printf("%f ", MAT_AT(m, i, j));
    }
    printf("\n");
  }
}

void mat_rand(Mat m, float low, float high) 
{
  for(size_t i = 0; i < m.rows; ++i) {
    for(size_t j = 0; j < m.cols; ++j) {
      MAT_AT(m, i, j) = rand_float()*(high - low) + low;
    }
  }
}

void mat_fill(Mat m, float x)
{
  for(size_t i = 0; i < m.rows; ++i) {
    for(size_t j = 0; j < m.cols; ++j) {
      MAT_AT(m, i, j) = x;
    }
  }
}

#endif //NL_H_