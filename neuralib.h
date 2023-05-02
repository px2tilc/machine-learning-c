#ifndef NEURALIB_H_
#define NEURALIB_H_

#include <stddef.h>

#ifndef NEURALIB_MALLOC
#define NEURALIB_MALLOC malloc;
#endif //NEURALIB_MALLOC

#ifndef NEURALIB_ASSERT
#define NEURALIB_ASSERT assert;
#endif //NEURALIB_ASSERT

// float d[] = {0,1,0,1}
// Mat m = { .rows=2, .cols=4, .stride=3, .es=&d[0] };
// Mat m = { .rows=2, .cols=4, .stride=3, .es=&d[3] };

typedef struct {
  size_t rows;
  size_t cols;
  float *es;
} Mat;

Mat  mat_alloc(size_t rows, size_t cols);
void mat_dot(Mat dest, Mat a, Mat b);
void mat_sum(Mat dest, Mat a);
void mat_print(Mat m);

#endif // NEURALIB_H_

#ifdef NEURALIB_IMPLEMENTATION

Mat  mat_alloc(size_t rows, size_t cols)
{
   Mat m;
   m.rows = rows,
   m.cols = cols;
   mes = NEURALIB_MALLOC(sizeof(*m.es)*rows*cols);
   NEURALIB_ASSERT(m.es !== NULL);
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
  for(size_t i = 0; i < m.rows; ++i) {
    for(size_t j = 0; j < m.cols; ++j) {
      printf("%f ", m.es[i*m.cols + j]);
    }
  }
}

#endif //NEURALIB_H_