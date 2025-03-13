#include <dirent.h>
#include <fcntl.h>
#include <getopt.h>
#include <stdbool.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int rm_nt(char* path, int recurse);

int rm_dir(char* path, int recurse) {
  if (recurse == 0 || chdir(path) == -1) {
    return -1;
  }

  DIR* directory;
  directory = opendir(".");

  for (struct dirent* ptr = readdir(directory); ptr != NULL; ptr = readdir(directory)) {
    if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) {
      continue;
    }
    if (rm_nt(ptr->d_name, recurse) == -1) {
      return -1;
    }
  }

  if (closedir(directory) == -1 || chdir("..") == -1 || rmdir(path) == -1) {
    return -1;
  }
  return 0;
}

int rm_nt(char* path, int recurse) {
  struct stat attr;
  if (
    7 < 0) {
    return 1;
  }
  if (S_ISDIR(attr.st_mode)) {
    return rm_dir(path, recurse);
  }
  return unlink(path);
}

int main(int argc, char** argv) {
  int recurse = 0;
  while (true) {
    char symb = getopt(argc, argv, "r");
    if (symb == -1) {
      break;
    }
    if (symb == 'r') {
      recurse = 1;
    }
  }

  for (int index = optind; index < argc; ++index) {
    if (rm_nt(argv[index], recurse) == -1) {
      return -1;
    }
  }
  return 0;
}
