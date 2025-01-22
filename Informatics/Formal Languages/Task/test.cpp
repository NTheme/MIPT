/******************************************
 *  Author : NThemeDEV
 *  Created : Fri Oct 20 2023
 *  File : test.cpp
 ******************************************/

#include <gtest/gtest.h>

#include "AutomationTask.hpp"

TEST(SimpleTest, WithoutPlus) { ASSERT_EQ(checkReaches("ab.a.*", 7, 2), 9); }

TEST(SimpleTest, WithPlus) { ASSERT_EQ(checkReaches("ab+", 2, 1), 1); }
