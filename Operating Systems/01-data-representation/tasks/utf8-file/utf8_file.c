#include "utf8_file.h"

#include <fcntl.h>
#include <malloc.h>
#include <errno.h>
#include <unistd.h>

int utf8_write(utf8_file_t* f, const uint32_t* str, size_t count) {
  for (size_t ind = 0; ind < count; ++ind) {
    uint8_t cur_log = 0;
    for (; str[ind] >> cur_log != 0; ++cur_log) {
    }
    uint8_t cur_len = (cur_log > 7) ? (cur_log + 3) / 5 : 1;

    uint8_t cur_sym = 0;
    for (uint8_t byte = 0; byte < cur_len; ++byte) {
      if (cur_len == 1) {
        cur_sym = str[ind];
      } else if (byte == 0) {
        cur_sym = (((1 << cur_len) - 1) << (8 - cur_len)) + ((str[ind] >> (6 * (cur_len - 1))));
      } else {
        cur_sym = (1 << 7) + (str[ind] >> (6 * (cur_len - byte - 1)) & ((1 << 6) - 1));
      }

      ssize_t res = write(f->fd, &cur_sym, 1);
      if (res == -1) {
        errno = 5;
        return -1;
      }
    }
  }
  
  return count;
}

int utf8_read(utf8_file_t* f, uint32_t* str, size_t count) {
  uint8_t cur_len = 0;
  uint8_t cur_sym = 0;
  size_t done = 0;

  while (done < count) {
    ssize_t res = read(f->fd, &cur_sym, 1);
    if (res == -1 || (res == 0 && cur_len != 0)) {
      errno = 5;
      return -1;
    }
    if (res == 0 && cur_len == 0) {
      return done;
    }

    if (cur_len == 0) {
      for (; cur_len < 8; ++cur_len) {
        if ((cur_sym >> (8 - cur_len - 1) & 1) == 0) {
          break;
        }
      }

      if (cur_len > 6) {
        errno = 27;
        return -1;
      }

      if (cur_len == 0) {
        *(str + done++) = cur_sym;
      } else if (cur_len < 8) {
        *(str + done) = (cur_sym & ((1 << (8 - cur_len - 1)) - 1)) << (6 * (cur_len - 1));
        --cur_len;
      }
    } else {
      *(str + done) += (cur_sym & ((1 << 7) - 1)) << (6 * (cur_len - 1));
      if (--cur_len == 0) {
        ++done;
      }
    }
  }

  return done;
}

utf8_file_t* utf8_fromfd(int fd) {
  utf8_file_t* file = NULL;
  file = (utf8_file_t*)calloc(sizeof(utf8_file_t), 1);
  if (file != NULL) {
    file->fd = fd;
  }
  return file;
}

/*
0-7
8-11
12-16
17-21
22-26
27-31
32

*/