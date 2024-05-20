#ifndef SORT
#define SORT

/**
 * @brief Sorting an array
 *
 * @param [in] mas An array to sort
 * @param [in] low First element to sort (counting since 0)
 * @param [in] high Last element to sort (size - 1)
 * @param [in] size Size of each element (sizeof)
 * @param [in] comp Comparator
 * @return void
 */
void qsort_rec(void *mas, int low, int high, size_t size, int comp(const void *, const void *));

#endif