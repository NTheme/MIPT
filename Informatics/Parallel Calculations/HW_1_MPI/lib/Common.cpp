/******************************************\
 *  Author  : NTheme - All rights reserved
 *  Created : 04 October 2024, 7:58 PM
 *  File    : Common.cpp
 *  Project : PD-1
\******************************************/

#include "../include/Common.hpp"

double integrate(double (*func)(double), double lhs, double rhs, long long num_steps) {
  double result = 0;
  double step = (rhs - lhs) / static_cast<double>(num_steps);

  for (int i = 0; i < num_steps - 1; ++i) {
    double x = lhs + step * i;
    result += step * (func(x) + func(x + step)) / 2;
  }

  return result;
}

double func(double x) {
  return 4 / (1 + x * x);
}
