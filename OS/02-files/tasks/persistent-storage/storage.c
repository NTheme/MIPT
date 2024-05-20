#include "storage.h"

#include <fcntl.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

char buffer_storage[65536];

int open_nt(storage_t* storage, storage_key_t key, int create) {
  char* pos = buffer_storage;
  strcpy(pos, storage->root_path);
  pos += strlen(storage->root_path);

  size_t ind = 0;
  while (ind + SUBDIR_NAME_SIZE <= strlen(key)) {
    *pos = '/';
    strncpy(++pos, key + ind, SUBDIR_NAME_SIZE);
    pos += SUBDIR_NAME_SIZE;
    ind += SUBDIR_NAME_SIZE;
    *pos = '\0';

    mkdir(buffer_storage, 0777);
    if (access(buffer_storage, F_OK) == -1 && create == 0) {
      return -1;
    }
  }
  
  *pos = '/';
  strcpy(++pos, ind == strlen(key) ? "\1" : key + ind);
  return open(buffer_storage, O_RDWR | (create == 1 ? O_CREAT : 0), 0666);
}

void storage_init(storage_t* storage, const char* root_path) {
  storage->root_path = (char*)malloc(strlen(root_path) + 1);
  strcpy(storage->root_path, root_path);
}

void storage_destroy(storage_t* storage) { free(storage->root_path); }

version_t storage_set(storage_t* storage, storage_key_t key, storage_value_t value) {
  int fd = open_nt(storage, key, 1);
  version_t shift = lseek(fd, 0, SEEK_END) / MAX_VALUE_SIZE + 1;
  memset(buffer_storage, 0, MAX_VALUE_SIZE);
  strcpy(buffer_storage, value);
  if (write(fd, buffer_storage, MAX_VALUE_SIZE) != MAX_VALUE_SIZE) {
    exit(1);
  }
  close(fd);
  return shift;
}

version_t storage_get(storage_t* storage, storage_key_t key, returned_value_t returned_value) {
  int fd = open_nt(storage, key, 0);
  if (fd == -1) {
    exit(0);
  }

  off_t shift = lseek(fd, -MAX_VALUE_SIZE, SEEK_END) / MAX_VALUE_SIZE + 1;
  if (read(fd, returned_value, MAX_VALUE_SIZE) != MAX_VALUE_SIZE) {
    exit(1);
  }
  close(fd);
  return shift;
}

version_t storage_get_by_version(storage_t* storage, storage_key_t key, version_t version,
                                 returned_value_t returned_value) {
  int fd = open_nt(storage, key, 0);
  lseek(fd, (version - 1) * MAX_VALUE_SIZE, SEEK_SET);
  if (read(fd, returned_value, MAX_VALUE_SIZE) == -1) {
    exit(1);
  }
  close(fd);
  return version;
}
