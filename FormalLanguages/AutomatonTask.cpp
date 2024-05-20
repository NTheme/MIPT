/******************************************
 *  Author : NThemeDEV
 *  Created : Mon Oct 16 2023
 *  File : AutomatonTask.cpp
 ******************************************/

#include <iostream>

#include "AutomationTask.hpp"

signed main() {
  std::string regular;
  size_t k, l;
  std::cin >> regular >> k >> l;
  std::cout << checkReaches(regular, k, l);
  return 0;
}
