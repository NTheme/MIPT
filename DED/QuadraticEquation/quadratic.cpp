#include <stdio.h>
#include <math.h>

#include "common.h"
#include "test.h"
#include "quadratic.h"

namespace quadratic {
    /**
     * @brief Solving linear equation
     * @param b  - Coefficient of x
     * @param c  - Free coefficient
     * @param x1 - Root
     * @return number of roots to num_roots (look ROOT_NUMBER)
     */
    static int solve_linear(long double b, long double c, long double *x1);

    /**
     * @brief Check if Equation is valid without errors
     * @param *equation - Checked Equation
     * @return 0 if vavlid and QE_QUAD_ERROR if not
     */
    inline int equation_valid(const Equation *equation) {
        if (equation == NULL || !isfinite(equation->a) || !isfinite(equation->b) || !isfinite(equation->c) || !isfinite(equation->x1) || !isfinite(equation->x2))
            return QE_QUAD_ERROR;
        return 0;
    }

    Equation *make_equation(long double a, long double b, long double c, int num_roots, long double x1, long double x2) {
        Equation *equation = (Equation *)calloc(1, sizeof(Equation));
        ASSERTIF(!equation_valid(equation), "unable to alloc", NULL);
        
        equation->a  = a;
        equation->b  = b;
        equation->c  = c;
        equation->x1 = x1;
        equation->x2 = x2;
        equation->num_roots = num_roots;
        return equation;
    }

    Equation **realloc_equations(Equation **equations, size_t size) {
        Equation **new_equations = (Equation **)realloc(equations, size * sizeof(Equation *));
        ASSERTIF(new_equations != NULL, "nullptr in equations", NULL);
        
        return new_equations;
    }

    void free_equations(Equation **equations, int quantity) {
        if(equations == NULL)
            return;
        
        for (int i = 0; i < quantity; i++) {
            free(equations[i]);
        }
        free(equations);
    }

    int equation_stream_input(Equation **equation, void *param, QUADRATIC_DEBUG test) {
        ASSERTIF(param   != NULL, "nullptr in stream",   0);
        ASSERTIF(equation != NULL, "nullptr in equation", 0);

        FILE *stream = (FILE *)param;

        if(test == QD_NDEBUG) {
            long double a = NAN, b = NAN, c = NAN;
            if (fscanf(stream, "%Lf %Lf %Lf", &a, &b, &c) != 3) {
                return 0;
            }

            *equation = make_equation(a, b, c);
        } else if (test == QD_DEBUG) {
            long double a = NAN, b = NAN, c = NAN, x1 = NAN, x2 = NAN;
            int num_roots = quadratic::RN_DEFAULT;
            if (fscanf(stream, "%Lf %Lf %Lf %d %Lf %Lf", &a, &b, &c, &num_roots, &x1, &x2) != 6) {
                return 0;
            }

            *equation = make_equation(a, b, c, num_roots, x1, x2);
        } else {
            printf("Unknown input mode\n");
            return 0;
        }
        
        return 1;
    }

    int equation_terminal_input(Equation **equation, const char **input) {
        ASSERTIF(input    != NULL, "nullptr in argv",     0);
        ASSERTIF(equation != NULL, "nullptr in equation", 0);

        long double a = NAN, b = NAN, c = NAN;
        if (sscanf(*input, "%Lf", &a) + sscanf(*(input + 1), "%Lf", &b) + sscanf(*(input + 2), "%Lf", &c) != 3) {
            return 0;
        }

        *equation = make_equation(a, b, c);
        return 1;
    }

    int stream_input(Equation ***equations, FILE *stream, int start_index, QUADRATIC_DEBUG test) {
        ASSERTIF(stream    != NULL, "nullptr in stream",    0);
        ASSERTIF(equations != NULL, "nullptr in equations", 0);

        size_t size = (size_t)start_index + 1;
        *equations = realloc_equations(*equations, size);

        Equation *equation = NULL;
        int read = 0;
        for (; equation_stream_input(&equation, (void *)stream, test) == 1; ++read) {
            if (read + start_index == (int)size) {
                *equations = realloc_equations(*equations, (size *= 2));
            }
            (*equations)[read + start_index] = equation;
        }
        *equations = realloc_equations(*equations, (size_t)(read + 1));

        return read;    
    }

    int terminal_input(Equation ***equations, int argc, const char **argv) {
        ASSERTIF(argv      != NULL, "nullptr in argv",      0);
        ASSERTIF(equations != NULL, "nullptr in equations", 0);

        Equation **tests = NULL;

        int file_flag = 1, num_equations = 0, num_tests = 0;
        for (; file_flag < argc - 1; file_flag += 2) {
            if (argv[file_flag][0] != '-') {
                break;
            }

            FILE *input = fopen(argv[file_flag + 1], "r");
            if (input == NULL) {
                printf("Wrong name filename %s\n", argv[file_flag + 1]);
            } else {                    
                switch (argv[file_flag][1])
                {
                case 't':
                    num_tests += stream_input(&tests, input, num_tests, QD_DEBUG);
                    break;
                case 'f':
                    num_equations += stream_input(equations, input, num_equations, QD_NDEBUG);
                    break;             
                default:
                    printf("Unknown flag %c", argv[file_flag][1]);
                    break;
                }
                fclose(input);
            }
        }

        if (tests != NULL) {
            unit_tests::test_quadratic(tests, num_tests);
        }

        size_t size = (size_t)num_equations + 1;
        *equations = realloc_equations(*equations, size);

        Equation *equation = NULL;
        int read = 0;
        for (; file_flag + 2 < argc && equation_terminal_input(&equation, argv + file_flag) == 1; ++read, file_flag += 3) {
            if (read + num_equations == (int)size) {
                *equations = realloc_equations(*equations, (size *= 2));
            }
            (*equations)[read + num_equations] = equation;
        }
       *equations = realloc_equations(*equations, (size_t)(read + num_equations + 1));
        return read + num_equations;
    }

    int print_roots(const Equation *equation) {
        ASSERTIF(!equation_valid(equation), "QUAD_ERROR in equation", QE_QUAD_ERROR);

        switch (equation->num_roots) {
        case RN_INF:
            printf("infinity of roots");
            break;
        case RN_ZERO:
            printf("zero roots");
            break;
        case RN_ONE:
            printf("one  root:  %+-10.5Lg", equation->x1);
            break;
        case RN_TWO:
            printf("two  roots: %+-10.5Lg %+-10.5Lg", equation->x1, equation->x2);
            break;
        case RN_DEFAULT:
            printf("uninitialized");
            break;
        default:
            ASSERTIF(0, "default case", 1);
        }

        return 0;
    }

    static int solve_linear(long double b, long double c, long double *x1) {
        ASSERTIF(x1 != NULL, "nullptr in x1", 1);

        if (common::is_zero(b)) {
            return (common::is_zero(c)) ? RN_INF : RN_ZERO;
        }

        *x1 = -c / b;
        return RN_ONE;
    }

    int solve_equation(Equation *equation) {
        ASSERTIF(!equation_valid(equation), "QUAD_ERROR in equation", QE_QUAD_ERROR);

        long double a = equation->a, b = equation->b, c = equation->c;

        if (common::is_zero(a)) {
            return solve_linear(b, c, &equation->x1);
        }

        long double discriminant = b * b - 4 * a * c;
        if (discriminant < 0)
            return RN_ZERO;

        if (common::is_zero(discriminant)) {
            equation->x1 = -b / (2 * a);
            return RN_ONE;
        }

        discriminant = sqrtl(discriminant);
        equation->x1 = (-b + discriminant) / (2 * a);
        equation->x2 = (-b - discriminant) / (2 * a);
        return RN_TWO;
    }
}
