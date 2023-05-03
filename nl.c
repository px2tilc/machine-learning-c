#define NL_IMPLEMENTATION
#include "nl.h"
#include <time.h>

typedef struct {
  // Input matrix
  Mat a0, a1, a2;
  // Layer 1
  Mat w1, b1;
  // Layer 2
  Mat w2, b2;
} Xor;

Xor xor_alloc(void)
{
    Xor m;
    // Input matrix
    m.a0 = mat_alloc(1, 2);  
    // Layer 1: Prepare weights and biases as matrices
    m.w1 = mat_alloc(2, 2);
    m.b1 = mat_alloc(1, 2);

    // Matrix for layer 1 processing / activations
    m.a1 = mat_alloc(1, 2);

    // Layer 2: Prepare weights and biases as matrices
    m.w2 = mat_alloc(2, 1);
    m.b2 = mat_alloc(1, 1);

    // Matrix for layer 2 processing / activations
    m.a2 = mat_alloc(1, 1);
    return m; 
}

void forward_xor(Xor m)
{
  // Layer 1 processing
  mat_dot(m.a1, m.a0, m.w1);
  mat_sum(m.a1, m.b1);
  mat_sig(m.a1);

  // Layer 2 processing
  mat_dot(m.a2, m.a1, m.w2);
  mat_sum(m.a2, m.b2);
  mat_sig(m.a2);
}

// Mat ti - matrix of train input data
// Mat to - matrix of train output data
float cost(Xor m, Mat ti, Mat to)
{
  NL_ASSERT(ti.rows == to.rows);
  NL_ASSERT(to.cols == m.a2.cols)
  size_t n  = ti.rows;

  float cost = 0.0f;
  for(size_t i = 0; i < n; ++i) {
    Mat x = mat_row(ti, i);
    // Expected
    Mat y = mat_row(to, i);
    
    mat_copy(m.a0, x);
    forward_xor(m);

    size_t q  = to.cols;
    for(size_t j = 0; j < q; j++) {
      float diff = MAT_AT(m.a2, 0, j) - MAT_AT(y, 0, j);
      cost += diff*diff;
    }
  }

  return cost/n;
}

void finite_diff(Xor m, Xor g, float eps, Mat ti, Mat to) 
{
  float saved;

  float c = cost(m, ti, to);

  for(size_t i = 0; i < m.w1.rows; ++i) {
    for(size_t j = 0; j < m.w1.cols; ++j) {
      saved = MAT_AT(m.w1, i, j);
      MAT_AT(m.w1, i, j) += eps;
      MAT_AT(g.w1, i, j) = (cost(m, ti, to) - c) / eps;
      MAT_AT(m.w1, i, j) = saved;
    }
  }

  for(size_t i = 0; i < m.b1.rows; ++i) {
    for(size_t j = 0; j < m.b1.cols; ++j) {
      saved = MAT_AT(m.b1, i, j);
      MAT_AT(m.b1, i, j) += eps;
      MAT_AT(g.b1, i, j) = (cost(m, ti, to) - c) / eps;
      MAT_AT(m.b1, i, j) = saved;
    }
  }

  for(size_t i = 0; i < m.w2.rows; ++i) {
    for(size_t j = 0; j < m.w2.cols; ++j) {
      saved = MAT_AT(m.w2, i, j);
      MAT_AT(m.w2, i, j) += eps;
      MAT_AT(g.w2, i, j) = (cost(m, ti, to) - c) / eps;
      MAT_AT(m.w2, i, j) = saved;
    }
  }

  for(size_t i = 0; i < m.b2.rows; ++i) {
    for(size_t j = 0; j < m.b2.cols; ++j) {
      saved = MAT_AT(m.b2, i, j);
      MAT_AT(m.b2, i, j) += eps;
      MAT_AT(g.b2, i, j) = (cost(m, ti, to) - c) / eps;
      MAT_AT(m.b2, i, j) = saved;
    }
  }
}

// Learn
void xor_learn(Xor m, Xor g, float rate)
{
for(size_t i = 0; i < m.w1.rows; ++i) {
    for(size_t j = 0; j < m.w1.cols; ++j) {
      MAT_AT(m.w1, i, j) -= rate*MAT_AT(g.w1, i, j);
    }
  }

  for(size_t i = 0; i < m.b1.rows; ++i) {
    for(size_t j = 0; j < m.b1.cols; ++j) {
      MAT_AT(m.b1, i, j) -= rate*MAT_AT(g.b1, i, j);
    }
  }

  for(size_t i = 0; i < m.w2.rows; ++i) {
    for(size_t j = 0; j < m.w2.cols; ++j) {
      MAT_AT(m.w2, i, j) -= rate*MAT_AT(g.w2, i, j);
    }
  }

  for(size_t i = 0; i < m.b2.rows; ++i) {
    for(size_t j = 0; j < m.b2.cols; ++j) {
      MAT_AT(m.b2, i, j) -= rate*MAT_AT(g.w2, i, j);
    }
  }
}

// Training data
float td[] = {
  0, 0, 0,
  0, 1, 1,
  1, 0, 1,
  1, 1, 0,
};

int main()
{
  srand(time(0));

  size_t stride = 3;
  size_t n = sizeof(td) / sizeof(td[0]) / stride;
  
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
  
  Xor m = xor_alloc();
  Xor g = xor_alloc();

  // 'Fill' all with random values
  mat_rand(m.w1, 0, 1);
  mat_rand(m.b1, 0, 1);
  mat_rand(m.w2, 0, 1);
  mat_rand(m.b2, 0, 1);
  
  float eps = 0.1;
  float rate = 0.1;

  for(size_t i = 0; i < 1000*100; ++i) {
    finite_diff(m, g, eps, ti, to);
    xor_learn(m, g, rate);
    printf("cost = %f\n", cost(m, ti, to));
  }

  printf("---------------------------------------\n");

  for(size_t i = 0; i < 2; ++i) {
    for(size_t j = 0; j < 2; ++j) {
      MAT_AT(m.a0, 0, 0) = i;
      MAT_AT(m.a0, 0, 1) = j;    
      forward_xor(m);
      // Output - result 
      float y = *m.a2.es;
      if (y < 0.5) {
        y = 0;
      } else {
        y = 1; 
      }
      printf("%zu ^ %zu = %f\n", i, j, y);
    }
  }

  return 0;
}