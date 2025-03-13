#ifndef PROCESS
#define PROCESS

#include <stdio.h>

typedef struct {
  const char *pointer;
  size_t len;
} text;

/**
 * @brief Counts number of symbols in file
 * @param [in] file File
 * @return Size of file
 */
size_t f_size(FILE *file);

/**
 * @brief Checks if a char is russian or english alpha
 *
 * @param [in] a A char to check
 * @return 1 if yes and 0 otherwise
 */
int is_alpha_rus(char a);

/**
 * @brief Compare 2 letters
 *
 * @param [in] a First letter
 * @param [in] b Second letter
 * @return 1 if a > b, -1 if a < b and 0 if a == b
 */
int compare_letters(char a, char b);

/**
 * @brief Checks if read string is not empty
 *
 * @param [in] string A read string
 * @param [in] num    Length of string
 * @return 1 if Ok and 0 otherwise
 */
int is_appropriate(const char *string, size_t len);

/**
 * @brief Counts strings in text (number of '\n' symbols)
 *
 * @param [in] book      A textstring
 * @param [in] file_size Size of text string
 * @return Number of strings
 */
size_t count_strings(const char *book, size_t file_size);

/**
 * @brief Counts number of spaces in the beginning of string
 *
 * @param [in] pt Pointer to first element
 * @return int Number of spaces
 */
size_t num_spaces(const char *pt);

/**
 * @brief Makes an array of text with strings
 *
 * @param [in]  book      Pointer to main string
 * @param [in]  file_size Size of book
 * @param [out] book_ret  Pointer to out text array
 * @return Number of strings
 */
size_t make_onegin(const char *book, size_t file_size, text **book_ret);

/**
 * @brief Prints all strings in text
 *
 * @param [in] ong     Source text
 * @param [in] num_str Number of strings
 * @param [in] stream  Output fluw
 */
void print_ong(const text *ong, size_t num_str, FILE *stream);

/**
 * @brief Makes a new copy of existing array of structures with strings
 *
 * @param [in] src  Pointer to array with text
 * @param [in] size Size of array
 * @return A pointer to copy of src
 */
text *onegin_cpy(text *src, size_t size);

#endif
