//
// Created by akhtyamovpavel on 5/1/20.
//

#include "LeapTestCase.h"

#include <Functions.h>

TEST(LeapTestCase, YearSimple) {
  EXPECT_EQ(false, IsLeap(1));
  EXPECT_EQ(true, IsLeap(4));
  EXPECT_EQ(false, IsLeap(100));
  EXPECT_EQ(true, IsLeap(400));
}

TEST(LeapTestCase, YearExcept) {
  EXPECT_THROW(
      {
        try {
          IsLeap(-10);
        } catch (std::invalid_argument& error) {
          EXPECT_STREQ("Year must be greater than 0", error.what());
          throw std::invalid_argument("Year must be greater than 0");
        }
      },
      std::invalid_argument);
}

TEST(LeapTestCase, MonthSimple) {
  EXPECT_EQ(31, GetMonthDays(1, 1));
  EXPECT_EQ(28, GetMonthDays(1, 2));
  EXPECT_EQ(30, GetMonthDays(1, 4));
  EXPECT_EQ(29, GetMonthDays(4, 2));
}

TEST(LeapTestCase, MonthExcept) {
  EXPECT_THROW(
      {
        try {
          GetMonthDays(4, -10);
        } catch (std::invalid_argument& error) {
          EXPECT_STREQ("Month should be in range [1-12]", error.what());
          throw std::invalid_argument("Month should be in range [1-12]");
        }
      },
      std::invalid_argument);
}
