#ifndef QUADR_DEF
#define QUADR_DEF

/**
 * @brief This namespace includes tools to process and solve quadratic equation.
 */
namespace quadratic {
    /**
     * @brief   A struct with coefficients and roots of quadratic equation
     * @details Includes all information about equation, whole processing is made with Equation's values.
     * @param a        - Coefficient of x ^ 2
     * @param b        - Coefficient of x
     * @param c        - Free coefficient
     * @param numroots - Number of equation's roots.
     * @param x1       - First equation's root.
     * @param x2       - Second equation's root.
     */
    typedef struct {
        long double a, b, c;
        long double x1, x2;
        int num_roots;
    } Equation;

    /// Enumerated type of data consists of all different quantities of quadratic equation's roots. Type RM_DEFAULT is not a quantity. It is used for initialization.
    typedef enum {
        RN_DEFAULT, ///< Not initialized
        RN_ZERO,    ///< Zero roots
        RN_ONE,     ///< One root
        RN_TWO,     ///< Two roots
        RN_INF      ///< Infinity of roots
    } ROOT_NUMBER;

 
    /// Enumerated type of data with errors.
    typedef enum {
        QE_QUAD_ERROR = -1 ///< Error during initialization od Equation.
    } QUADRATIC_ERROR;

    /// A flag to distinguish tests and real equations
    typedef enum {
        QD_DEBUG, ///< Parse tests
        QD_NDEBUG ///< Parse input
    } QUADRATIC_DEBUG;

    /**
     * @brief Makes an Equation using it's elements as arguments.
     * @param [in] a        - Coefficient of x ^ 2
     * @param [in] b        - Coefficient of x
     * @param [in] c        - Free coefficient
     * @param [in] numroots - Number of equation's roots.
     * @param [in] x1       - First equation's root.
     * @param [in] x2       - Second equation's root.
     * @return A pointer to an equation was made
     */
    Equation *make_equation(long double a, long double b, long double c, int numroots = RN_DEFAULT, long double x1 = 0, long double x2 = 0);

    /**
     * @brief Reallocates memory for an array of pointers to Equation and returns new pointer.
     * @details Allocates memory dynamically.
     * @param [in] **equation - A pointer to the first element of array
     * @param [in] size       - New array's size (how much Quadratic* will be allocated, not in bytes)
     * @return A new pointer to array, already with size elements
     */
    Equation **realloc_equations(Equation **equations, size_t size);

    /**
     * @brief Function frees dynamic memory which was allocated for array of pointers to Quadratic.
     * @details
     * @param [in] equation - A pointer to the first element
     * @param [in] size     - Array's size (how much Quadratic* was allocated, not in bytes)
     * @return void
     */
    void free_equations(Equation **equation, int size); 

    /**
     * @brief Function reads an Equation from *stream.
     * @details By default *stream is equal to stdin, but you can change it by giving this function secaon argument. First must have type Equation ** because stream_input dynamically
     * allocates memory for equation and writes a pointer to result into first argument. If test == QD_NDEBUG, functionreads 3 long double numbers from stream and makes an Equation: a, b, c. 
     * If test == QD_DEBUG, function do same thing but with 6 numbers: a, b, c, num_roots, x1, x2.
     * @param [out] **equation - A pointer to pointer to equation. &equation will have a pointer to read equation
     * @param [in]  *stream    - A pointer to FILE. It is a pointer to stream with input data
     * @return 1 if equation was read successfully ant 0 otherwise (in case of any errors)
     */
    int equation_stream_input (Equation **equations, void *stream = (void *)stdin, QUADRATIC_DEBUG test = QD_NDEBUG);

    /**
     * @brief Function reads an Equation from array of strings.
     * @details First argument must have type Equation ** because stream_input dynamically allocates memory for equation and writes a pointer to result into first argument. 
     * Function reads 3 long double numbers from input, input + 1 and input + 2: a, b and c. Then makes an Equation.
     * @param [out] **equation - A pointer to pointer to equation. &equation will have a pointer to read equation
     * @param [in]  *argv      - Pointer to first string (where get a)
     * @return 1 if equation was read successfully ant 0 otherwise (in case of any errors)
     */
    int equation_terminal_input(Equation **equation, const char **input);

    /**
     * @brief Function reads a plenty of Equation from *stream.
     * @details By default *stream is equal to stdin, but you can change it by giving this function secaon argument. First must have type Equation *** because full_stream_input dynamically
     * allocates memory for array of Equation and writes a pointer to result into first argument. If test == QD_NDEBUG, function reads data from stream, each by 3 long double numbers in a row 
     * while new 3 numbers exists, last < 3 it throws away (it's not important how numbers located on lines): a, b and c. Parses by each 3 elements and uses equation_stream_input to make 
     * an Equation. If test == QD_DEBUG, function do same thing but with 6 numbers: a, b, c, num_roots, x1, x2.
     * @param [out] ***equation - A pointer to pointer to pointer to equation. &equation will have a pointer to an array of read equations
     * @param [in]  *stream     - A pointer to FILE. It is a pointer to stream with input data
     * @param [in]  start_index - Current size of equations 
     * @param [in]  test        - Mode of input
     * @return number of equations that was read successfully
     */
    int stream_input(Equation *** equations, FILE *stream = stdin, int start_index = 0, QUADRATIC_DEBUG test = QD_NDEBUG);

    /**
     * @brief Function reads a plenty of Equation from command line arguments (it can be a file)
     * @details First must have type Equation *** because full_stream_input dynamically allocates memory for array of Equation and writes a pointer to result into first argument. 
     * Function reads data from terminal arguments that was given to program. Function follows this algotihm: if element of argv is a flag -f, terminal_input tries to get plenty of Equation 
     * from file which name is in next element of argv using stream_input; if element of argv is a flag -t, terminal_input do same thing but assumes given equations in file as tests (6 numbers).
     * Same with each even index. If it is not possible, function reads data, each by 3 long double numbers in a row, one element from one element of argv, while new 
     * 3 numbers exists, last < 3 it throws away (it's not important how numbers located on lines): a, b and c. Starts from argv[1]. Parses by each 3 and makes an Equation.
     * @param [out] ***equation - A pointer to pointer to pointer to equation. &equation will have a pointer to an array of read equations
     * @param [in]  argc        - Number of terminal arguments
     * @param [in]  **argv      - Pointer to array with strings of terminal arguments
     * @return number of equations that was read successfully
     */
    int terminal_input (Equation *** equations, int argc, const char **argv);

    /**
     * @brief Solves a quadratic equation.
     * @details Writes long double roots to x1 and x2 (if they exists, otherwise 0), members of equation which pointer was given as first 
     * argument.
     * @param [in] *equation - A pointer to equation.
     * @return number of roots to num_roots (look ROOT_NUMBER), you must make equation->num_roots equal to this value
     */
    int solve_equation(Equation *equation);

    /**
     * @brief Prints roots of Equation
     * @details writes to stdout number of roots, then roots if they exists. Examples: infinity of roots, zero roots, one root: 'x1', two roots: 'x1', 'x2'
     * @param [in] *equation - A pointer to equation.
     * @return 0 if equation was written successfully ant 0 otherwise (in case of any errors)
     */
    int print_roots(const Equation *equation);
}

#endif
