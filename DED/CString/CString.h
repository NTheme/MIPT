#ifndef C_STD
#define C_STD

#include <assert.h>
#include <errno.h>
#include <stddef.h>
#include <stdio.h>

#ifndef _NDEBUGMY
#define ASSERTIF(cond, desc, val)                                                                  \
  {                                                                                                \
    assert((cond) && desc);                                                                        \
    if (!(cond)) {                                                                                 \
      printf("ERROR in %s with '%s' in line %d in function %s in file %s", #cond, #desc, __LINE__, \
             __PRETTY_FUNCTION__, __FILE__);                                                       \
      errno = -1;                                                                                  \
      return val;                                                                                  \
    }                                                                                              \
  }
#else
#define ASSERTIF(cond, desc, val)
#endif

namespace CString {
/**
 * @brief Write a string, followed by a newline, to stdout.
 * @return positive number if success and EOF otherwise
 */
int puts(const char *);

/**
 * @brief Find the first occurrence of symbol in string.
 * @param [in] istring A string where fuction tries to find a symbol
 * @param [in] symbol Symbol to find
 * @return A pointer to found symbol in case od success and NULL itherwise (including errors)
 */
char *strchr(const char *string, int symbol);

/**
 * @brief * Return the length of string.
 * @param [in] string String which size will be given
 * @return Size of string
 */
size_t strlen(const char *string);

/**
 * @brief Copy srcptr string to destptr string including ending symbol.
 * @param [out] destptr A destination string
 * @param [in]  srcptr  A source string
 * @return A pointer to destptr if success and NULL otherwise
 */
char *strcpy(char *destptr, const char *srcptr);

/**
 * @brief Copy srcptr string to destptr string including ending symbol but no more than num symbols.
 * @param [out] destptr A destination string
 * @param [in]  srcptr  A source string
 * @param [in]  num     Max quantity of symbols to copy
 * @return A pointer to destptr if success and NULL otherwise
 */
char *strncpy(char *destptr, const char *srcptr, size_t num);

/**
 * @brief  Append symbols from srcptr string onto destptr string.
 *
 * @param [out] destptr A destination string
 * @param [in]  srcptr  A source string
 * @return A pointer to destptr if success and NULL otherwise
 */
char *strcat(char *destptr, const char *srcptr);

/**
 * @brief  Append symbols from srcptr string onto destptr string but no more than num symbols.
 * @param [out] destptr A destination string
 * @param [in]  srcptr  A source string
 * @param [in]  num     Max quantity of symbols to copy
 * @return A pointer to destptr if success and NULL otherwise
 */
char *strncat(char *destptr, const char *srcptr, size_t num);

/**
 * @brief Get a newline-terminated string of finite length n from STREAM.
 * @param [out] string     Destination string
 * @param [in]  num        Length of string
 * @param [in]  filestream Source filestream
 * @return A pointer to string is success and NULL otherwise
 */
char *fgets(char *string, int num, FILE *filestream);

/**
 * @brief Duplicate str string, returning an identical malloc'd string.
 *
 * @param str A string to dublicate
 * @return    A pointer to dublicated string if success and NULL otherwise
 */
char *strdup(const char *str);
/**
 * @brief Read up to (and including) a new line from STREAM into *lineptr (and null-terminate it). *lineptr is a pointer
 * returned from malloc (or NULL), pointing to *n characters of space. It is reallocced as necessary.
 * @param [out] lineptr A pointer to output string
 * @param [out] n       A size of buffer
 * @param [in]  stream  Source stream
 * @return The number of characters read (not including the null terminator), or -1 on error or EOF.
 */
size_t getline(char **lineptr, size_t *n, FILE *stream);
}  // namespace CString

#endif
