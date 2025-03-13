#include <gtest/gtest.h>

#include "../A/index.h"
#include "../B/lib.h"

TEST(FuncTest, AllInput) {
  for (int a = -1e9; a < 1e9; a += 1e7) {
    ASSERT_EQ(a + a / 16, fun(a, a / 16));
  }
}

TEST(LibTest, CorrectArray) {
  int x = 0, y = 1;
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(x, arr[i]);
    y = x + y;
    x = y - x;
  }
}
