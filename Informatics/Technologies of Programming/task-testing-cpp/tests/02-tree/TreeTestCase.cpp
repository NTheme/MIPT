//
// Created by akhtyamovpavel on 5/1/20.
//

#include "TreeTestCase.h"

#include "Tree.h"

void TreeTestCase::MakeFile(const std::filesystem::path& path) {
  std::ofstream file(path);
  file.close();
}

void TreeTestCase::MakeDir(const std::filesystem::path& path) {
  std::filesystem::create_directory(path);
}

TEST(TreeTestCase, ExistExcept) {
  EXPECT_THROW(
      {
        try {
          GetTree(std::string("a"), false);
        } catch (const std::invalid_argument& error) {
          EXPECT_STREQ("Path not exist", error.what());
          throw std::invalid_argument("Path not exist");
        }
      },
      std::invalid_argument);
}

TEST(TreeTestCase, FileExcept) {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() / "filename.txt";
  EXPECT_THROW(
      {
        TreeTestCase::MakeFile(path.c_str());
        try {
          GetTree(std::string(path), true);
        } catch (const std::invalid_argument& error) {
          EXPECT_STREQ("Path is not directory", error.what());
          throw std::invalid_argument("Path is not directory");
        }
      },
      std::invalid_argument);
  std::filesystem::remove(path);
}

TEST(TreeTestCase, Filter) {
  std::filesystem::path path = std::filesystem::temp_directory_path() / "dir1";
  TreeTestCase::MakeDir(path);
  TreeTestCase::MakeDir(path / "dir2");
  TreeTestCase::MakeFile(path / "dir2" / "file.txt");
  TreeTestCase::MakeDir(path / "dir3");

  FilterEmptyNodes(GetTree(path, false), path);
  EXPECT_EQ(1, std::filesystem::exists(path));
  EXPECT_EQ(1, std::filesystem::exists(path / "dir2"));
  EXPECT_EQ(1, std::filesystem::exists(path / "dir2" / "file.txt"));
  EXPECT_EQ(0, std::filesystem::exists(path / "dir3"));

  std::filesystem::remove_all(path.c_str());
}

TEST(TreeTestCase, Equal) {
  std::filesystem::path path = std::filesystem::temp_directory_path() / "dirr1";
  TreeTestCase::MakeDir(path);
  TreeTestCase::MakeDir(path / "dirr2");
  TreeTestCase::MakeFile(path / "dirr2" / "file.txt");
  TreeTestCase::MakeDir(path / "dirr2" / "dirr2");
  TreeTestCase::MakeDir(path / "dirr3");
  TreeTestCase::MakeDir(path / "dirr3" / "dirr2");

  EXPECT_EQ(1, GetTree(path, 0) == GetTree(path, 0));
  EXPECT_EQ(0, GetTree(path, 0) == GetTree(path / "dirr2", 0));
  EXPECT_EQ(0, GetTree(path, 1) == GetTree(path, 0));
  EXPECT_EQ(1, GetTree(path / "dirr2" / "dirr2", 0) ==
                   GetTree(path / "dirr3" / "dirr2", 0));

  std::filesystem::remove_all(path.c_str());
}
