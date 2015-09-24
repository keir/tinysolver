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
//
// A tiny least squares solver using Levenberg-Marquardt, intended for solving
// small dense problems with low-latency. The implementation takes care to do
// all allocation up front, so that no memory is allocated during solving.
// This is especially useful when solving many similar problems; for example,
// inverse pixel distortion for every pixel on a grid.
//
// Implementation taken almost straight from:
//
// [1] K. Madsen, H. Nielsen, O. Tingleoff.
//     Methods for Non-linear Least Squares Problems.
//     http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3215/pdf/imm3215.pdf

#ifndef TINY_SOLVER_H_

#include <cmath>

#include "Eigen/Core"
#include "Eigen/LU"

namespace tinysolver {

template<typename Function,
         typename LinearSolver = Eigen::PartialPivLU<
           Eigen::Matrix<typename Function::Scalar,
                         Function::NUM_PARAMETERS,
                         Function::NUM_PARAMETERS> > >
class TinySolver {
 public:
  enum {
    NUM_RESIDUALS = Function::NUM_RESIDUALS,
    NUM_PARAMETERS = Function::NUM_PARAMETERS
  };
  typedef typename Function::Scalar Scalar;
  typedef typename Eigen::Matrix<Scalar, NUM_PARAMETERS, 1> Parameters;

  // TODO(keir): Some of these knobs can be derived from each other and
  // removed, instead of requiring the user to set them.
  enum Status {
    RUNNING,
    // Resulting solution may be OK to use.
    GRADIENT_TOO_SMALL,            // eps > max(J'*f(x))
    RELATIVE_STEP_SIZE_TOO_SMALL,  // eps > ||dx|| / ||x||
    ERROR_TOO_SMALL,               // eps > ||f(x)||
    HIT_MAX_ITERATIONS,

    // Numerical issues
    FAILED_TO_EVALUATE_COST_FUNCTION,
    FAILED_TO_SOLVER_LINEAR_SYSTEM,
  };

  struct SolverParameters {
    SolverParameters()
       : gradient_threshold(1e-16),
         relative_step_threshold(1e-16),
         error_threshold(1e-16),
         initial_scale_factor(1e-3),
         max_iterations(100) {}
    Scalar gradient_threshold;       // eps > max(J'*f(x))
    Scalar relative_step_threshold;  // eps > ||dx|| / ||x||
    Scalar error_threshold;          // eps > ||f(x)||
    Scalar initial_scale_factor;     // Initial u for solving normal equations.
    int    max_iterations;           // Maximum number of solver iterations.
  };

  struct Results {
    Scalar error_magnitude;     // ||f(x)||
    Scalar gradient_magnitude;  // ||J'f(x)||
    int    num_failed_linear_solves;
    int    iterations;
    Status status;
  };

  Status Update(const Function& function, const Parameters &x) {
    // TODO(keir): Handle false return from the cost function.
    function(&x(0), &error_(0), &jacobian_(0, 0));
    error_ = -error_;

    // This explicitly computes the normal equations, which is numerically
    // unstable. Nevertheless, it is often good enough and is fast.
    jtj_ = jacobian_.transpose() * jacobian_;
    g_ = jacobian_.transpose() * error_;
    if (g_.array().abs().maxCoeff() < params.gradient_threshold) {
      return GRADIENT_TOO_SMALL;
    } else if (error_.norm() < params.error_threshold) {
      return ERROR_TOO_SMALL;
    }
    return RUNNING;
  }

  Results solve(const Function& function, Parameters* x_and_min) {
    Parameters& x = *x_and_min;
    results.status = Update(function, x);

    Scalar u = Scalar(params.initial_scale_factor * jtj_.diagonal().maxCoeff());
    Scalar v = 2;

    int i;
    for (i = 0; results.status == RUNNING && i < params.max_iterations; ++i) {
      jtj_augmented_ = jtj_;
      jtj_augmented_.diagonal().array() += u;

      linear_solver_.compute(jtj_augmented_);
      dx_ = linear_solver_.solve(g_);
      bool solved = (jtj_augmented_ * dx_).isApprox(g_);
      if (solved) {
        if (dx_.norm() < params.relative_step_threshold * x.norm()) {
          results.status = RELATIVE_STEP_SIZE_TOO_SMALL;
          break;
        }
        x_new_ = x + dx_;
        // Rho is the ratio of the actual reduction in error to the reduction
        // in error that would be obtained if the problem was linear. See [1]
        // for details.
        // TODO: Error handling on user eval.
        function(&x_new_[0], &f_x_new_[0], NULL);
        Scalar rho((error_.squaredNorm() - f_x_new_.squaredNorm())
                   / dx_.dot(u * dx_ + g_));
        if (rho > 0) {
          // Accept the Gauss-Newton step because the linear model fits well.
          x = x_new_;
          results.status = Update(function, x);
          Scalar tmp = Scalar(2 * rho - 1);
          u = u*std::max(1 / 3., 1 - tmp * tmp * tmp);
          v = 2;
          continue;
        }
      } else {
        results.num_failed_linear_solves++;
      }
      // Reject the update because either the normal equations failed to solve
      // or the local linear model was not good (rho < 0). Instead, increase u
      // to move closer to gradient descent.
      u *= v;
      v *= 2;
    }
    if (results.status == RUNNING) {
      results.status = HIT_MAX_ITERATIONS;
    }
    results.error_magnitude = error_.norm();
    results.gradient_magnitude = g_.norm();
    results.iterations = i;
    return results;
  }

  SolverParameters params;
  Results results;

 private:
  // Preallocate everything, including temporary storage needed for solving the
  // linear system. This allows reusing the intermediate storage across solves.
  LinearSolver linear_solver_;
  Parameters dx_, x_new_, g_;
  Eigen::Matrix<Scalar, NUM_RESIDUALS, 1> error_, f_x_new_;
  Eigen::Matrix<Scalar, NUM_RESIDUALS, NUM_PARAMETERS> jacobian_;
  Eigen::Matrix<Scalar, NUM_PARAMETERS, NUM_PARAMETERS> jtj_, jtj_augmented_;
};

}  // namespace tinysolver

#endif  // TINY_SOLVER_H_
 
