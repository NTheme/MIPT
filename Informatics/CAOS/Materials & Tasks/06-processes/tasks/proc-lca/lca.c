#include "lca.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char* PROC = "/proc/";
const char* STAT = "/stat";

void pid_to_path(pid_t pid, char* path) {
  strcpy(path, PROC);
  assert(path != NULL);
  sprintf(path + strlen(path), "%d", pid);
  strcpy(path + strlen(path), STAT);
}

pid_t get_ppid(pid_t pid) {
  char buffer[MAX_TREE_DEPTH] = {};
  pid_to_path(pid, buffer);
  FILE* file = fopen(buffer, "r");

  for (uint8_t cnt = 0; cnt < 4; ++cnt) {
    assert(fscanf(file, "%s", buffer) != 0 && "Cannot read stat!");
  }
  fclose(file);
  return strtol(buffer, NULL, 0);
}

void get_pid_tree(pid_t pid, pid_t* tree) {
  tree[0] = pid;
  for (pid_t len = 1; tree[len - 1] != 0; ++len) {
    tree[len] = get_ppid(tree[len - 1]);
  }
}

pid_t find_lca(pid_t x, pid_t y) {
  pid_t tree_x[MAX_TREE_DEPTH] = {};
  pid_t tree_y[MAX_TREE_DEPTH] = {};
  get_pid_tree(x, tree_x);
  get_pid_tree(y, tree_y);

  uint64_t pt_y = 0;
  for (uint64_t pt_x = 0;; ++pt_x) {
    for (; tree_x[pt_x] < tree_y[pt_y]; ++pt_y) {
    }
    if (tree_x[pt_x] == tree_y[pt_y]) {
      return tree_x[pt_x];
    }
  }
  return 0;
}
