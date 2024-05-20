#include "ieee754_clf.h"

#include <stdint.h>
#include <string.h>

float_class_t classify(double x) {
  uint64_t copy = 0;
  memcpy(&copy, &x, 8);
  
  uint64_t sign = (copy >> 63);
  uint64_t exponent = (copy >> 52) & (~(sign << 11));
  uint64_t mantissa = copy & (~(exponent << 52)) & (~(sign << 63));

  if (sign == 0) {
    if (exponent == 0) {
      if (mantissa == 0) {
        return Zero;
      }
      return Denormal;
    }
    if (exponent == 2047) {
      if (mantissa == 0) {
        return Inf;
      }
      return NaN;
    }
    return Regular;
  }
  if (exponent == 0) {
    if (mantissa == 0) {
      return MinusZero;
    }
    return MinusDenormal;
  }
  if (exponent == 2047) {
    if (mantissa == 0) {
      return MinusInf;
    }
    return NaN;
  }

  return MinusRegular;
}
