#include <stdbool.h>
#include <stdio.h>
#include <sys/stat.h>

#include "fcntl.h"
#include "unistd.h"

bool is_same_file(const char* lhs_path, const char* rhs_path) {
  struct stat lhs_attr, rhs_attr;
  stat(lhs_path, &lhs_attr);
  stat(rhs_path, &rhs_attr);

  if (access(lhs_path, F_OK) != 0 || access(rhs_path, F_OK) != 0 || lhs_attr.st_dev != rhs_attr.st_dev || lhs_attr.st_ino != rhs_attr.st_ino) {
    return false;
  }
  return true;
}

int main(int argc, const char* argv[]) {
  if(argc != 3) {
    return -1;
  }
  if (is_same_file(argv[1], argv[2])) {
    printf("yes");
  } else {
    printf("no");
  }
  return 0;
}
