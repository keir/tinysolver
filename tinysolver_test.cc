// Copyright (c) 2015 Keir Mierle
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
//
// Author: mierle@gmail.com (Keir Mierle)

#include "tinysolver.h"

#include <algorithm>
#include <cmath>

#include "gtest/gtest.h"

namespace tinysolver {

typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Matrix<double, 4, 1> Vec4;

class F {
 public:
  typedef double Scalar;
  enum {
    // Can also be Eigen::Dynamic.
    NUM_PARAMETERS = 3,
    NUM_RESIDUALS = 4,
  };
  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
    double x = parameters[0];
    double y = parameters[1];
    double z = parameters[2];

    residuals[0] = x + 2*y + 4*z;
    residuals[1] = y * z;

    if (jacobian) {
      jacobian[0 * 3 + 0] = 1;
      jacobian[0 * 3 + 1] = 2;
      jacobian[0 * 3 + 2] = 4;
      jacobian[1 * 3 + 0] = 0;
      jacobian[1 * 3 + 1] = 1;
      jacobian[1 * 3 + 2] = 1;
    }
    return true;
  }
};

TEST(TinySolver, SimpleExample) {
  Vec3 x(0.76026643, -30.01799744, 0.55192142);
  F f;
  TinySolver<F> solver;
  solver.solve(f, &x);
  Vec3 expected_min_x(2, 5, 0);
  EXPECT_NEAR(expected_min_x(0), x(0), 1e-5);
  EXPECT_NEAR(expected_min_x(1), x(1), 1e-5);
  EXPECT_NEAR(expected_min_x(2), x(2), 1e-5);
}

}  // namespace tinysolver
