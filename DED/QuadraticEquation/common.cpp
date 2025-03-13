#include <stdio.h>
#include <math.h>

#include "common.h"

namespace COLORS {
    const char *T_REGULAR = "\033[0m";
    const char *T_ARTICLE = "\033[1;3m";
    const char *T_GREEN =   "\033[0m\033[30;42m";
    const char *T_WHITE =   "\033[0m\033[34;47m"; 
    const char *T_BLUE =    "\033[0m\033[30;46m";
    const char *T_RED =     "\033[0m\033[37;41m";
};

namespace common {
    int is_zero(long double val) {
        return fabsl(val) < EPS;
    }

    void clearbuffer() {
	    while (getchar() != '\n');
    }
}
