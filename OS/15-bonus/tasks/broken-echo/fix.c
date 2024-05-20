#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

void fix_broken_echo() {
  int ppid = getppid();

  char path1[4096] = {};
  sprintf(path1, "/proc/%d/fd/3", ppid);
  int fd1 = open(path1, O_RDONLY);
  assert(fd1 != -1 && "Cannot open!");

  char path2[4096] = {};
  sprintf(path2, "/proc/%d/fd/1", ppid);
  int fd2 = open(path2, O_RDONLY);
  assert(fd2 != -1 && "Cannot open!");

  char buffer[4096] = {};
  ssize_t bytesRead = 0;

  while ((bytesRead = read(fd1, buffer, sizeof(buffer))) > 0) {
    ssize_t bytesWritten = write(fd2, buffer, bytesRead);
    assert(bytesWritten >= 0 && "Cannot write!");
  }

  assert(bytesRead >= 0 && "Cannot read!");

}
