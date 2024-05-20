#include "sort.h"

#include <stdio.h>

#include "common.h"

void qsort_rec(void *arr, int low, int high, size_t size, int comp(const void *, const void *)) {
  if (low < high) {
    char *pivot = (char *)arr + size * (size_t)high;

    int i = low;
    int j = low;
    while (i <= high) {
      if (comp((void *)((char *)arr + (size_t)i * size), (void *)pivot) > 0) {
        i++;
      } else {
        SWAP((char *)arr + (size_t)i * size, (char *)arr + (size_t)j * size, size);
        i++;
        j++;
      }
    }
    int pos = j - 1;

    qsort_rec(arr, low, pos - 1, size, comp);
    qsort_rec(arr, pos + 1, high, size, comp);
  }
}