#include <stdio.h>
#include <math.h>

#include "common.h"
#include "test.h"

namespace unit_tests {
    /**
     * @brief Checks if roots of equations are equal
     * @param [in] *a - First Equation 
     * @param [in] *b - Second Equation
     * @return 1 if roots are equal, 0 if not and -1 if error 
     */
    static int is_equal_roots(const quadratic::Equation *a, const quadratic::Equation* b);

    static int is_equal_roots(const quadratic::Equation *a, const quadratic::Equation* b) {
        ASSERTIF(a != NULL, "nullptr in a", quadratic::QE_QUAD_ERROR);
        ASSERTIF(b != NULL, "nullptr in b", quadratic::QE_QUAD_ERROR);

        if (a->num_roots != b->num_roots) {
            return 0;
        }
        if (a->num_roots == 1 && !common::is_zero(a->x1 - b->x1)) {
            return 0;
        }

        return (common::is_zero(a->x1 - b->x1) && common::is_zero(a->x2 - b->x2)) ||
               (common::is_zero(a->x1 - b->x2) && common::is_zero(a->x2 - b->x1));
    }

    int test_quadratic(quadratic::Equation **tests, int num_tests) {
        ASSERTIF(tests != NULL, "error in filename", 0);
        
        printf("%s%sTesting proram...\n", COLORS::T_WHITE, COLORS::T_ARTICLE);

        printf("\n%sNumber of tests: %3d", COLORS::T_WHITE, num_tests);

        for (int curtest = 0; curtest < num_tests; ++curtest) {
            printf("%s\nTest %3d...", COLORS::T_WHITE, curtest + 1);

            quadratic::Equation cur_equation = *tests[curtest];
            cur_equation.num_roots = quadratic::solve_equation(&cur_equation);

            switch (is_equal_roots(&cur_equation, tests[curtest])) {
            case quadratic::QE_QUAD_ERROR:
                free_equations(tests, num_tests);
                return 0;
            case 0:
                printf("%s%sWA!%s\n" "  %sEquation: ", COLORS::T_RED, COLORS::T_ARTICLE, COLORS::T_WHITE, COLORS::T_RED);
                printf(                       "a = %+.8Lg%s\n", tests[curtest]->a, COLORS::T_WHITE);
                printf("            %sb = %+.5Lg%s\n", COLORS::T_RED, tests[curtest]->b, COLORS::T_WHITE);
                printf("            %sc = %+.5Lg%s\n", COLORS::T_RED, tests[curtest]->c, COLORS::T_WHITE);
                printf("  \033[37;41mProgram answer: \033[30;46m");
                quadratic::print_roots(&cur_equation);
                printf("\033[34;47m\n  \033[37;41mCorrect answer: \033[30;46m");
                quadratic::print_roots(tests[curtest]);
                break;
            case 1:
                printf("%s%sOK!%s", COLORS::T_RED, COLORS::T_ARTICLE, COLORS::T_WHITE);
                break;
            default:
                ASSERTIF(0, "default case", 1);
            }
        }

        free_equations(tests, num_tests);
        printf("\n");
        return 0;
    }
}
