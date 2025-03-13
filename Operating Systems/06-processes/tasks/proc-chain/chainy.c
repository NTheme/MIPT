#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

enum { MAX_ARGS_COUNT = 256, MAX_CHAIN_LINKS_COUNT = 256, MAX_RETURN_VAL_COUNT = 256000 };

typedef struct {
  uint64_t argc;
  char* argv[MAX_ARGS_COUNT];
} chain_link_t;

typedef struct {
  uint64_t chain_links_count;
  chain_link_t chain_links[MAX_CHAIN_LINKS_COUNT];
} chain_t;

void create_link(char* command, chain_link_t* link) {
  uint64_t len = strlen(command);

  link->argc = 0;
  for (uint64_t ind = 0; ind < len; ++ind, ++link->argc) {
    for (; ind < len && command[ind] == ' '; ++ind) {
    }
    if (ind == len) {
      break;
    }

    uint64_t lhs = ind;
    for (; ind < len && command[ind] != ' '; ++ind) {
    }
    command[ind] = '\0';
    link->argv[link->argc] = command + lhs;
  }
}

void create_chain(char* command, chain_t* chain) {
  uint64_t len = strlen(command);

  chain->chain_links_count = 0;
  for (uint64_t ind = 0; ind < len; ++ind, ++chain->chain_links_count) {
    uint64_t lhs = ind;
    for (; command[ind] != '\0' && command[ind] != '|'; ++ind) {
    }
    command[ind] = '\0';
    create_link(command + lhs, chain->chain_links + chain->chain_links_count);
  }
}

void run_chain(chain_t* chain) {
  int stream[2][2];

  assert(pipe(stream[0]) != -1 && "Cannot create prev stream!");
  for (uint64_t i = 0; i < chain->chain_links_count; ++i) {
    assert(pipe(stream[1]) != -1 && "Cannot create curr stream!");

    if (fork() == 0) {
      for (uint8_t cnt = 0; cnt < 2; ++cnt) {
        dup2(stream[cnt][cnt], cnt);
        close(stream[0][cnt]);
        close(stream[1][cnt]);
      }
      execvp(chain->chain_links[i].argv[0], chain->chain_links[i].argv);
      exit(EXIT_SUCCESS);
    } else {
      for (uint8_t cnt = 0; cnt < 2; ++cnt) {
        close(stream[0][cnt]);
        stream[0][cnt] = stream[1][cnt];
      }
    }
  }

  char output[MAX_RETURN_VAL_COUNT];
  read(stream[1][0], output, MAX_RETURN_VAL_COUNT);
  printf("%s", output);
}

int main(int argc, char* argv[]) {
  assert(argc == 2 && "Invalit arguments number!");

  chain_t chain;
  create_chain(argv[1], &chain);
  run_chain(&chain);

  return 0;
}
