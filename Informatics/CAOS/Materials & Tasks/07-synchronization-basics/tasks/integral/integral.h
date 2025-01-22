#pragma once

#include <assert.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/param.h>

#include "wait.h"

typedef double field_t;

typedef field_t func_t(field_t);

enum STATUS { CALC, WAIT, STOP };
const field_t DX = 1e-5;

typedef struct thread_info {
  _Atomic uint32_t status;
  pthread_t thread;
  func_t* func;
  field_t lhs;
  field_t rhs;
  field_t res;
} thread_info_t;

typedef struct par_integrator {
  uint64_t threads_count;
  thread_info_t* threads;
} par_integrator_t;

void* thread_func(void* argv) {
  thread_info_t* info = (thread_info_t*)argv;

  while (true) {
    atomic_wait(&info->status, WAIT);
    uint32_t status = atomic_load(&info->status);

    if (status == WAIT) {
      continue;
    }
    if (status == STOP) {
      break;
    }

    info->res = 0;
    for (uint64_t ind = 0; info->lhs + DX * ind < info->rhs; ++ind) {
      field_t lhs = info->lhs + DX * ind;
      field_t rhs = MIN(info->lhs + DX * (ind + 1), info->rhs);
      info->res += (rhs - lhs) * info->func((rhs + lhs) / 2);
    }

    atomic_store(&info->status, WAIT);
    atomic_notify_all(&info->status);
  }

  return NULL;
}

int par_integrator_init(par_integrator_t* integrator, size_t threads_count) {
  integrator->threads_count = threads_count;
  integrator->threads = (thread_info_t*)calloc(integrator->threads_count, sizeof(thread_info_t));
  assert(integrator->threads != NULL && "Cannot allocate threads!");

  pthread_attr_t attr;
  pthread_attr_init(&attr);

  for (uint64_t ind = 0; ind < integrator->threads_count; ++ind) {
    atomic_store(&integrator->threads[ind].status, WAIT);
    assert(pthread_create(&integrator->threads[ind].thread, &attr, thread_func, &integrator->threads[ind]) == 0 &&
           "Cannot create a thread!");
  }
  pthread_attr_destroy(&attr);

  return 0;
}

int par_integrator_start_calc(par_integrator_t* integrator, func_t* func, field_t lhs, field_t rhs) {
  for (uint64_t ind = 0; ind < integrator->threads_count; ++ind) {
    integrator->threads[ind].func = func;
    integrator->threads[ind].lhs = lhs + (rhs - lhs) / integrator->threads_count * ind;
    integrator->threads[ind].rhs = lhs + (rhs - lhs) / integrator->threads_count * (ind + 1);

    atomic_store(&integrator->threads[ind].status, CALC);
    atomic_notify_one(&integrator->threads[ind].status);
  }

  return 0;
}

int par_integrator_get_result(par_integrator_t* integrator, field_t* res) {
  *res = 0;
  for (uint64_t ind = 0; ind < integrator->threads_count; ++ind) {
    atomic_wait(&integrator->threads[ind].status, CALC);
    *res += integrator->threads[ind].res;
  }

  return 0;
}

int par_integrator_destroy(par_integrator_t* integrator) {
  for (uint64_t ind = 0; ind < integrator->threads_count; ++ind) {
    atomic_store(&integrator->threads[ind].status, STOP);
    atomic_notify_one(&integrator->threads[ind].status);
    pthread_join(integrator->threads[ind].thread, NULL);
  }

  free(integrator->threads);
  return 0;
}
