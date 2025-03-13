#ifndef COMMON
#define COMMON

#include <assert.h>
#include <errno.h>

#ifndef _NDEBUGMY
#define ASSERTIF(cond, desc, val)                                                                  \
  {                                                                                                \
    assert((cond) && desc);                                                                        \
    if (!(cond)) {                                                                                 \
      printf("ERROR in %s with '%s' in line %d in function %s in file %s", #cond, #desc, __LINE__, \
             __PRETTY_FUNCTION__, __FILE__);                                                       \
      errno = -1;                                                                                  \
      return val;                                                                                  \
    }                                                                                              \
  }
#else
#define ASSERTIF(cond, desc, val)
#endif

#define SWAP(a, b, size)         \
  do {                           \
    size_t __size = (size);      \
    char *__a = (a), *__b = (b); \
    do {                         \
      char __tmp = *__a;         \
      *__a++ = *__b;             \
      *__b++ = __tmp;            \
    } while (--__size > 0);      \
  } while (0)

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#endif
