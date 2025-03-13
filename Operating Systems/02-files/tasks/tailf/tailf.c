#include "fcntl.h"
#include "stdbool.h"
#include "stdio.h"
#include "sys/stat.h"
#include "unistd.h"

void tail_out(int fd) {
  char b;
  while (read(fd, &b, 1)) {
    printf("%c", b);
  }
}

int main(int argc, char** argv) {
  int fd = open(argv[1], O_RDONLY);

  long size_o = -1;
  while (true) {
    struct stat attr;
    stat(argv[1], &attr);
    long size_n = attr.st_size;

    if (size_o != size_n) {
      tail_out(fd);
      fflush(stdout);
      size_o = size_n;
    }
  }

  close(fd);
  return 0;
}
