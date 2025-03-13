
#include <mpi.h>

#include "include/Common.hpp"
#include "include/MPICalc.hpp"

int main(int argc, char **argv) {
  int world_rank = init(&argc, &argv);

  for (int num_parts : NUM_PARTS) {
    for (int num_procs = 1; num_procs <= MAX_PROCS; ++num_procs) {
      if (world_rank == 0) {
        mainProcess(num_parts, num_procs);
      } else if (world_rank < num_procs) {
        descProcess(num_parts);
      }
    }
  }

  finalize();
  return 0;
}
