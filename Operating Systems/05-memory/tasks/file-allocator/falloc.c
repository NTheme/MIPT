#include "falloc.h"

#include <assert.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

void falloc_init(file_allocator_t* allocator, const char* filepath, uint64_t allowed_page_count) {
  int fd = -1;
  if (access(filepath, F_OK) == 0 && allocator->allowed_page_count == allowed_page_count) {
    fd = open(filepath, O_RDWR);
    assert(fd != -1 && "Can't open a file!");
  } else if (access(filepath, F_OK) != 0) {
    fd = open(filepath, O_RDWR | O_CREAT, 0666);
    assert(fd != -1 && "Can't create a file!");

    for (uint64_t cnt = 0; cnt < PAGE_SIZE * (allowed_page_count + 1); ++cnt) {
      ssize_t res_fd = write(fd, "\0", 1);
      assert(res_fd == 1 && "Unable to fill file!");
    }
  }

  allocator->fd = fd;
  allocator->allowed_page_count = allowed_page_count;

  int prot = PROT_READ | PROT_WRITE;
  int flag = MAP_SHARED | MAP_FILE;
  void* page_mask = mmap(0, PAGE_SIZE, prot, flag, fd, 0);
  void* base_addr = mmap(0, PAGE_SIZE * allowed_page_count, prot, flag, fd, PAGE_SIZE);
  assert(page_mask != MAP_FAILED && "Unable to mmap mask!");
  assert(base_addr != MAP_FAILED && "Unable to mmap base!");

  allocator->page_mask = (uint64_t*)page_mask;
  allocator->base_addr = base_addr;

  allocator->curr_page_count = 0;
  char* mask = (char*)page_mask;
  for (uint64_t ind = 0; ind < allowed_page_count; ++ind) {
    allocator->curr_page_count += mask[ind] - '\0';
  }
}

void falloc_destroy(file_allocator_t* allocator) {
  int res_base = munmap(allocator->base_addr, PAGE_SIZE * allocator->allowed_page_count);
  int res_mask = munmap(allocator->page_mask, PAGE_SIZE);
  assert(res_base != -1 && "Unable to unmap base!");
  assert(res_mask != -1 && "Unable to unmap mask!");

  int res_fd = close(allocator->fd);
  assert(res_fd != -1 && "unable to close file!");

  allocator->base_addr = NULL;
  allocator->page_mask = NULL;
  allocator->curr_page_count = 0;
}

void* falloc_acquire_page(file_allocator_t* allocator) {
  if (allocator->curr_page_count == allocator->allowed_page_count) {
    return NULL;
  }
  char* mask = (char*)allocator->page_mask;

  uint64_t ind = 0;
  for (; mask[ind] == 1; ++ind) {
  }
  mask[ind] = '\1';

  ++allocator->curr_page_count;
  return allocator->base_addr + PAGE_SIZE * ind;
}

void falloc_release_page(file_allocator_t* allocator, void** addr) {
  if (*addr - allocator->base_addr < 0 || *addr - allocator->base_addr >= PAGE_SIZE * allocator->allowed_page_count ||
      (*addr - allocator->base_addr) % PAGE_SIZE != 0) {
    return;
  }

  char* page = (char*)(*addr);
  for (uint64_t shift = 0; shift < PAGE_SIZE; ++shift) {
    page[shift] = '\0';
  }

  char* mask = (char*)allocator->page_mask;
  mask[(*addr - allocator->base_addr) / 4096] = '\0';

  --allocator->curr_page_count;
  *addr = NULL;
}
