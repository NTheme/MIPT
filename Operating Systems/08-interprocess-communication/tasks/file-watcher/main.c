#include <assert.h>
#include <linux/limits.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

typedef struct Counter {
  char filename[PATH_MAX];
  int counter;
  struct Counter* next;
} Counter;

typedef struct Counters {
  struct Counter* head;
} Counters;

void increment(Counters* counters, char* filename, int value) {
  Counter* current = counters->head;
  while (current != NULL) {
    if (strncmp(current->filename, filename, PATH_MAX) == 0) {
      current->counter += value;
      return;
    }
    current = current->next;
  }
  Counter* new_head = (Counter*)malloc(sizeof(Counter));
  new_head->next = counters->head;
  new_head->counter = value;
  strcpy(new_head->filename, filename);
  counters->head = new_head;
}

void print(Counters* counters) {
  Counter* current = counters->head;
  while (current != NULL) {
    printf("%s:%d\n", current->filename, current->counter);
    current = current->next;
  }
}

bool wait_syscall(pid_t pid) {
  while (true) {
    ptrace(PTRACE_SYSCALL, pid, NULL, NULL);
    int status;
    waitpid(pid, &status, 0);

    if (WIFSTOPPED(status) && (WSTOPSIG(status) & 128) != 0) {
      return false;
    }
    if (WIFEXITED(status)) {
      return true;
    }
  }
}

void fd_to_path(pid_t pid, int fd, char* path) {
  char buf[PATH_MAX];
  sprintf(buf, "/proc/%d/fd/%d", pid, fd);
  path[readlink(buf, path, PATH_MAX)] = '\0';
}

int main(int argc, char* const* argv) {
  assert(argc > 1 && "No args given!");

  Counters* counters = (Counters*)malloc(sizeof(Counter));
  counters->head = NULL;

  pid_t pid = fork();
  assert(pid != -1 && "Cannot create fork!");

  if (pid == 0) {
    ptrace(PTRACE_TRACEME);
    kill(getpid(), SIGSTOP);
    execvp(argv[1], argv + 1);
    exit(EXIT_FAILURE);
  } else {
    waitpid(pid, NULL, 0);
    ptrace(PTRACE_SETOPTIONS, pid, NULL, PTRACE_O_TRACESYSGOOD);

    while (true) {
      struct __ptrace_syscall_info syscall_info;
      if (wait_syscall(pid)) {
        break;
      }
      assert(ptrace(PTRACE_GET_SYSCALL_INFO, pid, sizeof(syscall_info), &syscall_info) != -1 &&
             "PTrace syscall error!");

      if (syscall_info.op != PTRACE_SYSCALL_INFO_ENTRY || syscall_info.entry.nr != 1) {
        continue;
      }

      char filepath[PATH_MAX];
      fd_to_path(pid, syscall_info.entry.args[0], filepath);

      if (wait_syscall(pid)) {
        break;
      }
      assert(ptrace(PTRACE_GET_SYSCALL_INFO, pid, sizeof(syscall_info), &syscall_info) != -1 &&
             "PTrace syscall error!");

      if (syscall_info.op != PTRACE_SYSCALL_INFO_EXIT) {
        continue;
      }

      increment(counters, filepath, syscall_info.exit.rval);
    }
  }

  print(counters);
  return 0;
}
