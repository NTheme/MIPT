#include <stdio.h>
#include <iostream>

#include "quadratic.h"
#include "common.h"
#include "test.h"

int main(int argc, const char *argv[]) {
	printf("%s%s# Program for solving quadratic equations a * x^2 + b * x + c = 0\n", COLORS::T_BLUE, COLORS::T_ARTICLE);
	printf("# By NThemeDEV (c) 2022 ver. 0.9%s\n", COLORS::T_GREEN);

	quadratic::Equation **equations = NULL;
	int numequations = quadratic::terminal_input(&equations, argc, argv);

	char inputmore = 0;
	if (argc > 1) {		
		printf("%s%d equations read from the terminal arguments (also file). Do you want to input manually (y or n)? ", COLORS::T_GREEN, numequations);
		while (scanf("%c", &inputmore) != 1 || (inputmore != 'y'  && inputmore != 'n')) {
			printf("%sWrong input! Try again:%s ", COLORS::T_RED, COLORS::T_GREEN);
			common::clearbuffer();
		}
	}

	if (inputmore == 'y' || argc == 1) {
		int add = 0;
		printf("Input number of equations: ");
		while (scanf("%d", &add) != 1 || add < 0) {
			printf("%sWrong input! Try again:%s ", COLORS::T_RED, COLORS::T_GREEN);
			common::clearbuffer();
		}

		equations = quadratic::realloc_equations(equations, (size_t)(numequations + add));
		for (int i = 0; i < add; i++) {
			printf("Input a, b, c of %d your extra equation divided by space:%s ", i + 1, COLORS::T_GREEN);
			if (equation_stream_input(&equations[numequations + i]) != 1) {
				printf("%sWrong input! Try again. ", COLORS::T_RED);
				common::clearbuffer();
				i--;
			}
		}
		numequations += add;
	}

	for (int i = 0; i < numequations; i++) {
		printf("%s\nEquation %3d with a = %+-10.5Lg b = %+-10.5Lg and c = %+-10.5Lg ", COLORS::T_GREEN, i + 1, equations[i]->a, equations[i]->b, equations[i]->c);

		if ((equations[i]->num_roots = quadratic::solve_equation(equations[i])) == quadratic::QE_QUAD_ERROR) {
			printf("%sis unable to be solved!%s\n", COLORS::T_RED, COLORS::T_GREEN);
		} else {
			printf("has %s", COLORS::T_BLUE);
			print_roots(equations[i]);
		}
	}

	printf("%s%s\n", COLORS::T_GREEN, COLORS::T_REGULAR);

	free_equations(equations, numequations);
	return 0;
}
