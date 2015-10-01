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

typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::VectorXd VecX;

bool EvaluateResidualsAndJacobians(const double* parameters,
                                   double* residuals,
                                   double* jacobian) {
  double x = parameters[0];
  double y = parameters[1];
  double z = parameters[2];

  residuals[0] = x + 2*y + 4*z;
  residuals[1] = y * z;

  if (jacobian) {
    jacobian[0 * 2 + 0] = 1;
    jacobian[0 * 2 + 1] = 0;

    jacobian[1 * 2 + 0] = 2;
    jacobian[1 * 2 + 1] = z;

    jacobian[2 * 2 + 0] = 4;
    jacobian[2 * 2 + 1] = y;
  }
  return true;
}

class ExampleStatic {
 public:
  typedef double Scalar;
  enum {
    // Can also be Eigen::Dynamic.
    NUM_PARAMETERS = 3,
    NUM_RESIDUALS = 2,
  };
  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
    return EvaluateResidualsAndJacobians(parameters, residuals, jacobian);
  }
};

class ExampleParametersDynamic {
 public:
  typedef double Scalar;
  enum {
    NUM_PARAMETERS = Eigen::Dynamic,
    NUM_RESIDUALS = 2,
  };

  int NumParameters() const {
    return 3;
  }

  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
    return EvaluateResidualsAndJacobians(parameters, residuals, jacobian);
  }
};

class ExampleResidualsDynamic {
 public:
  typedef double Scalar;
  enum {
    NUM_PARAMETERS = 3,
    NUM_RESIDUALS = Eigen::Dynamic,
  };

  int NumResiduals() const {
    return 2;
  }

  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
    return EvaluateResidualsAndJacobians(parameters, residuals, jacobian);
  }
};

class ExampleAllDynamic {
 public:
  typedef double Scalar;
  enum {
    NUM_PARAMETERS = Eigen::Dynamic,
    NUM_RESIDUALS = Eigen::Dynamic,
  };

  int NumParameters() const {
    return 3;
  }

  int NumResiduals() const {
    return 2;
  }

  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
    return EvaluateResidualsAndJacobians(parameters, residuals, jacobian);
  }
};

template <typename Function, typename Vector>
void TestHelper(const Function& f, const Vector& x0) {
  Vector x = x0;
  Vec2 residuals;
  f(x.data(), residuals.data(), NULL);
  EXPECT_GT(residuals.norm(), 1e-10);

  TinySolver<Function> solver;
  solver.solve(f, &x);

  f(x.data(), residuals.data(), NULL);
  EXPECT_NEAR(0.0, residuals.norm(), 1e-10);
}

// A test case for when the cost function is statically sized.
TEST(TinySolver, SimpleExample) {
  Vec3 x0(0.76026643, -30.01799744, 0.55192142);
  ExampleStatic f;

  TestHelper(f, x0);
}


// A test case for when the number of parameters is dynamically sized.
TEST(TinySolver, ParametersDynamic) {
  VecX x0(3);
  x0 << 0.76026643, -30.01799744, 0.55192142;

  ExampleParametersDynamic f;

  TestHelper(f, x0);
}

// A test case for when the number of residuals is dynamically sized.
TEST(TinySolver, ResidualsDynamic) {
  Vec3 x0(0.76026643, -30.01799744, 0.55192142);

  ExampleResidualsDynamic f;

  TestHelper(f, x0);
}

// A test case for when the number of parameters and residuals is
// dynamically sized.
TEST(TinySolver, ParametersAndResidualsDynamic) {
  VecX x0(3);
  x0 << 0.76026643, -30.01799744, 0.55192142;

  ExampleAllDynamic f;

  TestHelper(f, x0);
}

}  // namespace tinysolver
