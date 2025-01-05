<!--========================================   <
    * Author  : NTheme - All rights reserved
    * Created : 16 December 2024, 4:23 AM
    * File    : README
    * Project : Salesman
>   ======================================== -->

Salesman
===

An implementation of 1.5-approximation algorithm for Salesman problem.

Build & Install
---

**1. Clone the repo and go to project folder**

  ```sh
  git clone https://github.com/NTheme/Salesman.git
  cd Salesman
  ```

**2. Now you are ready to build and install.**

- Make sure you are in the root project directory.
- Set `CMAKE_INSTALL_PREFIX` to the folder where you want to install the application (for example `.` or `/usr/bin`).
  Binary files will appear in the `CMAKE_INSTALL_PREFIX/bin` directory.
- Set `flows` to the number of your processor's cores increased by 1.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=. -G"Unix Makefiles"
cmake --build build --config Release -- -j <flows>
sudo cmake --install build
```

Launch
---

Execute the following command from the root project directory.

```sh
CMAKE_INSTALL_PREFIX/bin/ComplexityProject <flags>
```

Usage
---

Follow the instructions inside the application.

Author
---

✨ ***NTheme*** ✨  
All rights reserved ©
