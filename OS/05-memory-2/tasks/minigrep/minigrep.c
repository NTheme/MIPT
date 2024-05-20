#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <linux/limits.h>
#include <pcre.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

void process_file(char* path, pcre* regular, off_t file_size) {
  int fd = open(path, O_RDONLY);
  assert(fd != -1 && "Cannot open file!");

  char* content = mmap(NULL, file_size, PROT_READ, MAP_FILE | MAP_PRIVATE, fd, 0);
  assert(content != MAP_FAILED && "Cannot mmap!");

  for (off_t start = 0, line_num = 1; start < file_size; ++line_num) {
    off_t end = start;
    for (; end < file_size && content[end] != '\n'; ++end) {
    }

    int match[3];
    if (pcre_exec(regular, NULL, content + start, end - start, 0, 0, match, 3) >= 0) {
      printf("%s:%ld: %.*s\n", path, line_num, (int)(end - start), content + start);
    }

    start = end + 1;
  }

  assert(munmap(content, file_size) == 0 && "Cannot unmap!");
  assert(close(fd) == 0 && "Cannot close file!");
}

void process_dir(char* path, pcre* regular) {
  size_t path_length = strlen(path);

  DIR* dir = opendir(path);
  assert(dir != NULL && "Cannot open dir!");

  for (struct dirent* dirent_ptr = readdir(dir); dirent_ptr != NULL; dirent_ptr = readdir(dir)) {
    if (strcmp(dirent_ptr->d_name, ".") == 0 || strcmp(dirent_ptr->d_name, "..") == 0) {
      continue;
    }
    sprintf(path + path_length, "/%s", dirent_ptr->d_name);

    struct stat st;
    assert(stat(path, &st) != -1 && "Cannot get stat!");

    if (S_ISDIR(st.st_mode)) {
      process_dir(path, regular);
    }
    process_file(path, regular, st.st_size);
  }
  path[path_length] = '\0';

  assert(closedir(dir) == 0 && "Cannot close dir!");
}

int main(int argc, char* argv[]) {
  assert(argc > 2 && "Invalid arguments!");

  int pos = 0;
  const char* msg = NULL;
  pcre* regular = pcre_compile(argv[1], 0, &msg, &pos, NULL);
  if (regular == NULL) {
    fprintf(stderr, "%s %d\n", msg, pos);
    assert(regular != NULL && "Cannot compile regex!");
  }

  process_dir(argv[2], regular);

  pcre_free(regular);
  return 0;
}
