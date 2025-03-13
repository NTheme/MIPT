#define _GNU_SOURCE

#include <assert.h>
#include <signal.h>
#include <unistd.h>

int outstrem;

void signal_handler(int sig) { assert(write(outstrem, &sig, sizeof(int)) != -1 && "Cannot handle!"); }

int signalfd() {
  int stream[2];
  assert(pipe(stream) != -1 && "Cannot create a stream!");
  outstrem = stream[1];

  for (int sig = 0; sig < _NSIG; ++sig) {
    signal(sig, signal_handler);
  }
  
  return stream[0];
}
