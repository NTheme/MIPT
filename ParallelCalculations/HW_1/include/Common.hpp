/******************************************\
 *  Author  : NTheme - All rights reserved
 *  Created : 04 October 2024, 7:57 PM
 *  File    : Common.hpp
 *  Project : PD-1
\******************************************/

#pragma once

static constexpr const char* OUTPUT = "result.txt";

static constexpr int NUM_PARTS[3] = {(int)1e3, (int)1e6, (int)1e8};
static constexpr int MAX_PROCS = 8;
static constexpr int PRECISION = 6;

static constexpr double LHS = 0;
static constexpr double RHS = 1;

double integrate(double (*func)(double), double lhs, double rhs, long long num_steps);

double func(double x);
