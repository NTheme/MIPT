#include "process.h"

#include <ctype.h>
#include <malloc.h>

#include "common.h"

size_t f_size(FILE *file) {
  ASSERTIF(file != 0, "nullptr in file", (size_t)0);

  fseek(file, 0, SEEK_END);
  size_t ret = (size_t)ftell(file);
  rewind(file);
  return ret;
}

int is_alpha_rus(char a) { return (a >= -64 && a <= -1) || a == -72 || a == -88 || isalpha(a); }

int compare_letters(char a, char b) {
  int a_1 = ((int)a % 32 + 32) % 32, b_1 = ((int)b % 32 + 32) % 32;

  if (a > 0) a_1 += 64;
  if (b > 0) b_1 += 64;

  if (a_1 > b_1) {
    return 1;
  }
  if (a_1 < b_1) {
    return -1;
  }
  return 0;
}

int is_appropriate(const char *string, size_t len) {
  ASSERTIF(string != NULL, "nullptr in str", (size_t)0);

  for (size_t i = 0; i < len; i++) {
    if (is_alpha_rus(*(string + i))) {
      return 1;
    }
  }
  return 0;
}

size_t count_strings(const char *book, size_t file_size) {
  ASSERTIF(book != NULL, "nullptr in book", (size_t)0);

  size_t num_str = 0;
  for (size_t add = 0; add < file_size; ++add) {
    if (book[add] == '\n') {
      ++num_str;
    }
  }

  return num_str;
}

size_t num_spaces(const char *pt) {
  ASSERTIF(pt != NULL, "nullptr in pt", (size_t)0);

  size_t space = 0;
  for (; *(pt + space) == ' '; ++space)
    ;

  return space;
}

size_t make_onegin(const char *book, size_t file_size, text **book_ret) {
  ASSERTIF(book != NULL, "nullptr in book", 0);

  size_t num_str = count_strings(book, file_size);
  *book_ret = (text *)calloc(num_str, sizeof(text));
  ASSERTIF(book_ret != NULL, "nullptr in book_ret", 0);

  size_t read = 0;
  int last_end = -1;
  for (size_t add = 0; add < file_size; ++add) {
    if (*(book + add) != '\n') {
      continue;
    }
    if (is_appropriate(book + last_end + 1, add - (size_t)last_end)) {
      size_t space = num_spaces(book + last_end + 1);
      (*book_ret)[read++] =
          (text){.pointer = book + (size_t)last_end + space + 1, .len = add - (size_t)last_end - space};
    }

    last_end = (int)add;
  }

  *book_ret = (text *)realloc(*book_ret, read * sizeof(text));

  return read;
}

void print_ong(const text *ong, size_t num_str, FILE *stream) {
  for (size_t str = 0; str < num_str; ++str) {
    for (size_t let = 0; let < ong[str].len; ++let) {
      fprintf(stream, "%c", *(ong[str].pointer + let));
    }
  }
}

text *onegin_cpy(text *src, size_t size) {
  text *origin = (text *)calloc(size, sizeof(text));
  ASSERTIF(origin != NULL, "nullptr in origin", NULL);

  for (size_t i = 0; i < size; i++) {
    origin[i] = (text){.pointer = src[i].pointer, .len = src[i].len};
  }

  return origin;
}
