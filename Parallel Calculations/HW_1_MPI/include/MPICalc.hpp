/******************************************\
 *  Author  : NTheme - All rights reserved
 *  Created : 04 October 2024, 9:13 PM
 *  File    : MPICalc.hpp
 *  Project : PD-1
\******************************************/

#pragma once

int init(int *argc, char ***argv);

void mainProcess(int num_parts, int num_processes);

void descProcess(int num_parts);

void finalize();
