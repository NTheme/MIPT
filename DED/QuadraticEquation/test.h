#ifndef TEST_DEF
#define TEST_DEF

#include "quadratic.h"

/**
 * @brief   This namespace includes helpful objects for testing solver of quadratic equations.
 * @details You must have a file with tests. If there is 1 root, put it as 5th number.
 */
namespace unit_tests {
    /**
     * @brief   This function tests the solver of quadratic equations.
     * @details Reads tests from a file with name from argument filename, each by 6 numbers in a row while new 6 numbers exists, last < 6 it throws away 
     * (it's not important how numbers located on lines): a, b, c, numroots, x1, x2. Then test_quadratic get answers by solver and compare them with correct numroots, x1 and x2.
     * In case of difference prints equation's coefficients, correct answers and given by the solver.
     * @param [in] *filename - name of file wiht tests
     * @return 0 If no errors happend and non-zero number otherwise
     */
    int test_quadratic(quadratic::Equation **equation, int num_tests);
}

#endif
