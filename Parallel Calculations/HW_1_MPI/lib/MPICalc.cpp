/******************************************\
 *  Author  : NTheme - All rights reserved
 *  Created : 04 October 2024, 9:14 PM
 *  File    : MPICalc.cpp
 *  Project : PD-1
\******************************************/

#include "../include/MPICalc.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "../include/Common.hpp"
#include "../include/Timer.hpp"
#include "mpi.h"

void printResult(const double *divided_result, double straight_result, double divided_time, double straight_time, int num_parts,
                 int num_processes) {
  std::cout << std::setprecision(PRECISION);
  std::ofstream out(OUTPUT, std::ios::app);

  std::cout << std::endl << "/****************************************************\\" << std::endl;
  std::cout << "Number of Parts: " << num_parts << ", Number of Processes: " << num_processes << std::endl;

  std::cout << std::endl << "-> Processes calculation:" << std::endl;
  double divided_sum = 0;
  for (int i = 0; i < num_processes; ++i) {
    std::cout << "---> Process " << i << " result:           " << divided_result[i] << std::endl;
    divided_sum += divided_result[i];
  }

  std::cout << std::endl << "-> Divided calculation result:   " << divided_sum << std::endl;
  std::cout << "-> Straight calculation result:  " << straight_result << std::endl;
  std::cout << "-> Acceleration:                 " << straight_time / divided_time << std::endl;
  std::cout << "\\****************************************************/\n" << std::endl << std::endl;
  out << num_parts << ' ' << num_processes << ' ' << straight_time / divided_time << std::endl;
}

void calculateDivided(int num_parts, int num_processes, double *result) {
  double step = (RHS - LHS) / static_cast<double>(num_processes);

  for (int process_index = 1; process_index < num_processes; ++process_index) {
    double bounds[] = {LHS + process_index * step, LHS + (process_index + 1) * step};
    MPI_Send(bounds, 2, MPI_DOUBLE_PRECISION, process_index, 0, MPI_COMM_WORLD);
  }

  result[0] = integrate(func, LHS, LHS + step, num_parts / num_processes);
  for (int slave_index = 1; slave_index < num_processes; ++slave_index) {
    MPI_Status status;
    MPI_Recv(result + slave_index, 1, MPI_DOUBLE_PRECISION, slave_index, 0, MPI_COMM_WORLD, &status);
  }
}

int init(int *argc, char ***argv) {
  MPI_Init(argc, argv);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0 && std::filesystem::exists(OUTPUT)) {
    std::filesystem::remove(OUTPUT);
  }

  return world_rank;
}

void mainProcess(int num_parts, int num_processes) {
  double divided_result[MAX_PROCS];

  nt::Timer timer;
  calculateDivided(num_parts, num_processes, divided_result);
  double divided_time = timer.get();

  timer.reset();
  double straight_result = integrate(func, LHS, RHS, num_parts);
  double straight_time = timer.get();

  printResult(divided_result, straight_result, divided_time, straight_time, num_parts, num_processes);
}

void descProcess(int num_parts) {
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  double bounds[2];
  MPI_Status status;
  MPI_Recv(bounds, 2, MPI_DOUBLE_PRECISION, 0, 0, MPI_COMM_WORLD, &status);

  double result = integrate(func, bounds[0], bounds[1], num_parts / world_size);
  MPI_Send(&result, 1, MPI_DOUBLE_PRECISION, 0, 0, MPI_COMM_WORLD);
}

void finalize() {
  std::cout.flush();
  MPI_Finalize();
}