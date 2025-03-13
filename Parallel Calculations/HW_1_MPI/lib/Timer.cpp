/******************************************\
 *  Author  : NTheme - All rights reserved
 *  Created : 04 October 2024, 7:33 PM
 *  File    : timer.cpp
 *  Project : PD-1
\******************************************/

#include "../include/Timer.hpp"

#include "mpi.h"

namespace nt {

Timer::Timer() {
  reset();
}

void Timer::reset() {
  start_time = MPI_Wtime();
}

double Timer::get() const {
  return MPI_Wtime() - start_time;
}

}  // namespace nt
