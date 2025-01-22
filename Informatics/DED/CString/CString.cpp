#include "CString.h"

#include <malloc.h>

int CString::puts(const char *string) {
  ASSERTIF(string != 0, "nullptr in string", EOF);

  do {
    putc(*string, stdout);
  } while (*string++);

  return 1;
}

char *CString::strchr(const char *string, int symbol) {
  ASSERTIF(string != 0, "nullptr in string", NULL);

  do {
    if (*string == ((char)symbol)) {
      return (char *)string;
    }
  } while (*string++);

  return NULL;
}

size_t CString::strlen(const char *string) {
  ASSERTIF(string != 0, "nullptr in string", 0);

  const char *pt = NULL;
  for (pt = string; *pt; ++pt)
    ;

  return (size_t)(pt - string);
}

char *CString::strcpy(char *destptr, const char *srcptr) {
  ASSERTIF(destptr != 0, "nullptr in destptr", NULL);
  ASSERTIF(srcptr != 0, "nullptr in srcptr", NULL);

  char *pt = destptr;
  while ((*pt++ = *srcptr++) != 0)
    ;

  return destptr;
}

char *CString::strncpy(char *destptr, const char *srcptr, size_t num) {
  ASSERTIF(srcptr != 0, "nullptr in srcptr", NULL);
  ASSERTIF(srcptr != 0, "nullptr in srcptr", NULL);

  char *pt = destptr;
  while (num) {
    if ((*pt = *srcptr) != 0) srcptr++;
    ++pt;
    --num;
  }

  return destptr;
}

char *CString::strcat(char *destptr, const char *srcptr) {
  ASSERTIF(srcptr != 0, "nullptr in srcptr", NULL);
  ASSERTIF(srcptr != 0, "nullptr in srcptr", NULL);

  char *pt = destptr;
  while (*pt++)
    ;
  --pt;
  while ((*pt++ = *srcptr++) != 0)
    ;

  return destptr;
}

char *CString::strncat(char *destptr, const char *srcptr, size_t num) {
  ASSERTIF(srcptr != 0, "nullptr in srcptr", NULL);
  ASSERTIF(srcptr != 0, "nullptr in srcptr", NULL);

  char *pt = destptr;
  while (*pt++)
    ;
  --pt;
  while (num && ((*pt = *srcptr++) != 0)) {
    --num, ++pt;
  }
  *pt = 0;

  return destptr;
}

char *CString::fgets(char *string, int num, FILE *filestream) {
  ASSERTIF(string != 0, "nullptr in string", NULL);
  ASSERTIF(filestream != 0, "nullptr in filestream", NULL);

  char *pt = string;

  int buf = '0';
  while (num && (buf = getc(filestream)) != '\n' && buf != EOF) *pt++ = (char)buf, --num;

  return string;
}

char *CString::strdup(const char *str) {
  ASSERTIF(str != 0, "nullptr in str", NULL);

  char *s = NULL;
  size_t size = (CString::strlen(str) + 1) * sizeof(char);

  if ((s = (char *)malloc(size)) != NULL) {
    CString::strcpy(s, str);
  }

  return s;
}

size_t CString::getline(char **lineptr, size_t *n, FILE *stream) {
  ASSERTIF(lineptr != 0, "nullptr in string", (size_t)-1);
  ASSERTIF(n != 0, "nullptr in string", (size_t)-1);
  ASSERTIF(stream != 0, "nullptr in string", (size_t)-1);

  char *s = NULL;
  size_t size = 1;
  s = (char *)malloc((size *= 2) * sizeof(char));

  size_t read = 0;
  int buf;
  while ((buf = getc(stream)) != '\n') {
    if (buf == EOF) {
      return (size_t)-1;
    }
    if (read == size) {
      s = (char *)realloc(s, (size *= 2) * sizeof(char));
    }
    s[read++] = (char)buf;
  }
  s[read++] = '\n';
  s[read] = '\0';
  ;

  if (*n < read) {
    *n = read;
    *lineptr = (char *)realloc(*lineptr, read * sizeof(char));
  }

  *lineptr = s;
  return read;
}

int main() {}
