#include <iostream>

size_t str_to_size(const char* ptr) {
  size_t num = 0;
  while (*ptr != '\0') {
    num = num * 10 + (*ptr++ - '0');
  }
  return num;
}

void alloc_memory(int**& arrays, size_t*& size_arrays, size_t num_arrays,
                  const char** argv) {
  arrays = new int* [num_arrays] {};
  size_arrays = new size_t[num_arrays]{};

  for (size_t i = 0; i < num_arrays; ++i) {
    size_arrays[i] = str_to_size(argv[i + 1]);
    arrays[i] = new int[size_arrays[i]]{};
  }
}

void free_memory(int** arrays, const size_t* size_arrays, size_t num_arrays) {
  for (size_t i = 0; i < num_arrays; ++i) {
    delete[] arrays[i];
  }
  delete[] size_arrays;
  delete[] arrays;
}

template <typename Type>
void fill(Type* array, size_t size, Type val) {
  for (size_t i = 0; i < size; ++i) {
    array[i] = val;
  }
}

bool check_permutation(const size_t* permutation, bool* exist_index,
                       size_t num_arrays, size_t max_array_size) {
  bool appropriate = true;
  fill(exist_index, max_array_size, false);

  for (size_t i = 0; i < num_arrays; ++i) {
    if (exist_index[permutation[i]]) {
      appropriate = false;
      break;
    }

    exist_index[permutation[i]] = true;
  }

  return appropriate;
}

bool get_next_permutation(size_t* permutation, const size_t* size_arrays,
                          size_t num_arrays, size_t max_array_size,
                          bool* exist_index) {
  do {
    for (size_t index = num_arrays; index > 0; --index) {
      permutation[index - 1]++;
      if (permutation[index - 1] < size_arrays[index - 1]) {
        break;
      }
      permutation[index - 1] = 0;

      if (index == 1) {
        return false;
      }
    }
  } while (
      !check_permutation(permutation, exist_index, num_arrays, max_array_size));
  return true;
}

long long get_permutation_value(int** arrays, const size_t* permutation,
                                size_t num_arrays) {
  long long sum = 1;
  for (size_t i = 0; i < num_arrays; ++i) {
    sum *= arrays[i][permutation[i]];
  }
  return sum;
}

size_t get_max_array_size(const size_t* size_arrays, size_t num_arrays) {
  size_t num = 0;
  for (size_t i = 0; i < num_arrays; ++i) {
    num = (num < size_arrays[i]) ? size_arrays[i] : num;
  }
  return num;
}

long long get_sum(int** arrays, size_t* size_arrays, size_t num_arrays) {
  long long sum = 0;
  size_t max_array_size = get_max_array_size(size_arrays, num_arrays);

  size_t* permutation = new size_t[num_arrays]{};
  bool* exist_index = new bool[max_array_size]{};

  for (size_t i = 0; get_next_permutation(permutation, size_arrays, num_arrays,
                                          max_array_size, exist_index);
       ++i) {
    sum += get_permutation_value(arrays, permutation, num_arrays);
  }

  delete[] permutation;
  delete[] exist_index;
  return sum;
}

int main(int argc, const char** argv) {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);

  size_t num_arrays = argc - 1;

  int** arrays;
  size_t* size_arrays;
  alloc_memory(arrays, size_arrays, num_arrays, argv);

  for (size_t i = 0; i < num_arrays; ++i) {
    for (size_t j = 0; j < size_arrays[i]; ++j) {
      std::cin >> arrays[i][j];
    }
  }

  std::cout << get_sum(arrays, size_arrays, num_arrays) << '\n';

  free_memory(arrays, size_arrays, num_arrays);
  return 0;
}
