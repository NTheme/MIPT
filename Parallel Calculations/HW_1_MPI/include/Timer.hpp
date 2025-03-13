/******************************************\
 *  Author  : NTheme - All rights reserved
 *  Created : 04 October 2024, 7:34 PM
 *  File    : timer.hpp
 *  Project : PD-1
\******************************************/

#pragma once

namespace nt {

class Timer {
 public:
  Timer();

  void reset();
  [[nodiscard]] double get() const;

 private:
  double start_time{};
};

}  // namespace nt
