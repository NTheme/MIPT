#include <locale.h>
#include <malloc.h>
#include <stdio.h>

#include "common.h"
#include "process.h"
#include "sort.h"

/**
 * @brief Comparator to straight sort
 *
 * @param [in] a First element to compare
 * @param [in] b Second element to compare
 * @return 1 if a > b, -1 if a < b and 0 if a == b
 */
int comp_straight(const void *a, const void *b);

/**
 * @brief Comparator to reversed sort
 *
 * @param [in] a First element to compare
 * @param [in] b Second element to compare
 * @return 1 if a > b, -1 if a < b and 0 if a == b
 */
int comp_reversed(const void *a, const void *b);

int comp_straight(const void *a, const void *b) {
  const char *str[2] = {(*(const text *)a).pointer, (*(const text *)b).pointer};
  size_t len[2] = {(*(const text *)a).len, (*(const text *)b).len};

  size_t p[2] = {0, 0};
  while (p[0] < len[0] && p[1] < len[1]) {
    for (int i = 0; i < 2; i++) {
      while (p[i] < len[i] && !is_alpha_rus(*(str[i] + p[i]))) {
        p[i]++;
      }
    }

    int rret = compare_letters(str[0][p[0]], str[1][p[1]]);
    if (rret > 0) {
      return 1;
    }
    if (rret < 0) {
      return -1;
    }
    p[0]++, p[1]++;
  }

  return 0;
}

int comp_reversed(const void *a, const void *b) {
  size_t len[2] = {(*(const text *)a).len - 1, (*(const text *)b).len - 1};
  const char *str[2] = {(*(const text *)a).pointer + len[0], (*(const text *)b).pointer + len[1]};

  size_t p[2] = {0, 0};
  while (p[0] < len[0] && p[1] < len[1]) {
    for (int i = 0; i < 2; i++) {
      while (p[i] < len[i] && !is_alpha_rus(*(str[i] - p[i]))) {
        p[i]++;
      }
    }

    int rret = compare_letters(*(str[0] - p[0]), *(str[1] - p[1]));
    if (rret > 0) {
      return 1;
    }
    if (rret < 0) {
      return -1;
    }
    p[0]++, p[1]++;
  }

  return 0;
}

int main() {
  setlocale(LC_ALL, "Russian");

  FILE *input = fopen("hamlet.txt", "r");
  ASSERTIF(input != NULL, "nullptr in input", 1);

  size_t file_size = f_size(input);

  char *book = (char *)calloc(file_size + 1, sizeof(char));
  ASSERTIF(book != NULL, "nullptr in book", 1);

  if (fread(book, sizeof(char), file_size, input) == file_size) {
    text *origin = NULL;
    size_t num_str = make_onegin(book, file_size, &origin);
    text *to_sort = onegin_cpy(origin, num_str);

    FILE *outpt = fopen("out.txt", "w");
    ASSERTIF(outpt != NULL, "nullptr in book", 1);

    fprintf(outpt, "---------------The Very HAMLET:-------------\n\n\n");
    qsort_rec(to_sort, 0, (int)num_str - 1, sizeof(text), comp_straight);
    print_ong(to_sort, num_str, outpt);

    fprintf(outpt, "\n\n---------------The Origin HAMLET:-------------\n\n\n");
    print_ong(origin, num_str, outpt);

    fprintf(outpt, "\n\n---------------The Very Reversed HAMLET:-------------\n\n\n");
    qsort_rec(to_sort, 0, (int)num_str - 1, sizeof(text), comp_reversed);
    print_ong(to_sort, num_str, outpt);

    free(to_sort);
    free(origin);
    fclose(outpt);
  }

  free(book);
  fclose(input);
  return 0;
}
