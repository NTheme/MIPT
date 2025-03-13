#include <getopt.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int mkdir_nt(char* path, int create, unsigned long mode) {
  char* ind = index(path, '/');
  if (ind == NULL) {
    return mkdir(path, mode);
  }

  char cur_path[4096];
  strncpy(cur_path, path, ind + 1 - path);
  cur_path[ind + 1 - path] = '\0';

  if (chdir(cur_path) == -1 && (create == 0 || mkdir(cur_path, 0777) == -1 || chdir(cur_path) == -1)) {
    return -1;
  }
  if (mkdir_nt(ind + 1, create, mode) == -1 || chdir("..") == -1) {
    return -1;
  }
  return 0;
}

int main(int argc, char** argv) {
  struct option modes[2] = {{.name = "mode", .has_arg = required_argument, .val = 'm', .flag = NULL}};

  int create_path = 0;
  unsigned long mode = 0777;
  while (true) {
    int symb = getopt_long(argc, argv, "pm:", modes, NULL);
    if (symb == -1) {
      break;
    }
    if (symb == 'p') {
      create_path = 1;
    } else if (symb == 'm') {
      mode = strtoul(optarg, NULL, 8);
    }
  }

  char cwd[4096];
  if (getcwd(cwd, 4096) == NULL) {
    return -1;
  }

  for (int i = optind; i < argc; ++i) {
    if (mkdir_nt(argv[i], create_path, mode) == -1 || chdir(cwd) == -1) {
      return -1;
    }
  }
  return 0;
}
