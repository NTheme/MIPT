#include "bloom_filter.h"

#include <malloc.h>
#include <string.h>

const uint64_t B_SIZE = sizeof(uint64_t) * 8;

uint64_t calc_hash(const char* str, uint64_t modulus, uint64_t seed) {
  uint64_t power = 37 + seed % modulus;

  uint64_t hash = 0;
  for (uint64_t index = 0; index < strlen(str); ++index) {
    hash = (hash * power + (str[index] - '\0')) % modulus;
  }
  return hash;
}

void bloom_init(struct BloomFilter* bloom_filter, uint64_t set_size,
                hash_fn_t hash_fn, uint64_t hash_fn_count) {
  bloom_filter->set = NULL;
  bloom_filter->set =
      (uint64_t*)calloc(B_SIZE / 8, (set_size + B_SIZE - 1) / B_SIZE);
  if (bloom_filter->set == NULL) {
    printf("Unable to allocate\n");
    return;
  }

  bloom_filter->set_size = set_size;
  bloom_filter->hash_fn = hash_fn;
  bloom_filter->hash_fn_count = hash_fn_count;
}

void bloom_destroy(struct BloomFilter* bloom_filter) {
  free(bloom_filter->set);
  bloom_filter->set = NULL;
}

void bloom_insert(struct BloomFilter* bloom_filter, Key key) {
  for (uint64_t iter = 0; iter < bloom_filter->hash_fn_count; ++iter) {
    uint64_t hash = bloom_filter->hash_fn(key, bloom_filter->set_size, iter);
    bloom_filter->set[hash / B_SIZE] |= (1UL << (hash % B_SIZE));
  }
}

bool bloom_check(struct BloomFilter* bloom_filter, Key key) {
  bool found = true;
  for (uint64_t iter = 0; iter < bloom_filter->hash_fn_count; ++iter) {
    uint64_t hash = bloom_filter->hash_fn(key, bloom_filter->set_size, iter);
    found = found && bloom_filter->set[hash / B_SIZE] & (1UL << (hash % B_SIZE));
  }
  return found;
}
