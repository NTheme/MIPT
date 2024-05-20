#pragma once

#include <assert.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct node {
  uintptr_t value;
  struct node* next;
} node_t;

typedef struct lfstack {
  _Atomic(node_t*) front;
} lfstack_t;

int lfstack_init(lfstack_t* stack) {
  atomic_init(&stack->front, NULL);
  return 0;
}

int lfstack_push(lfstack_t* stack, uintptr_t value) {
  node_t* next = (node_t*)calloc(1, sizeof(node_t));
  assert(next != NULL && "Cannot allocate new vertex!");

  next->value = value;
  next->next = NULL;

  node_t* curr = atomic_load(&stack->front);
  for (next->next = curr; !atomic_compare_exchange_weak(&stack->front, &curr, next);
       next->next = curr) {
  }
  return 0;
}

int lfstack_pop(lfstack_t* stack, uintptr_t* value) {
  node_t* curr = atomic_load(&stack->front);
  if (curr == NULL) {
    *value = 0;
    return 0;
  }

  for (node_t* next = curr->next; !atomic_compare_exchange_weak(&stack->front, &curr, next);
       next = curr->next) {
    if (curr == NULL) {
      *value = 0;
      return 0;
    }
  }
  *value = curr->value;

  free(curr);
  return 0;
}

int lfstack_destroy(lfstack_t* stack) { return 0; }
