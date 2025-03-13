//
// Created by akhtyamovpavel on 5/1/20.
//

#pragma once

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

class TreeTestCase : public ::testing::Test {
 public:
  static void MakeFile(const std::filesystem::path& path);
  static void MakeDir(const std::filesystem::path& path);
};
