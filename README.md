NOTE! This is not production just yet so I don't advise using it!

# Tiny Solver
A tiny least squares solver targeting extreme performance on repeated
minimizations of small dense problems.

## Why Tiny Solver?
What distinguishes Tiny Solver from other implementations of least squares
minimization is the following

* The user's cost function is directly inlined into the Levenberg-Marquardt
  minimization, eliminating function call overhead and exposing more
  optimization opportunities to the compiler.
* All memory is pre-allocated separately from the minimization, enabling
  repeated minimization with zero allocations.
* The code is short, simple, only depends on Eigen, is usable as a single
  header-only library (no separate compilation), and has a liberal license.

An example application of Tiny Solver is computing the camera warp inverse on a
per-pixel basis; in this case, the goal is to map each post-warp pixel back to
the source image. This involves doing a LM minimization for every pixel in a
potentially large (4K) image; in this case, Tiny Solver is ~15x faster than
Ceres Solver. Blender uses a predecessor of Tiny Solver for this task.

## How is Tiny Solver different than Ceres Solver?

* Ceres targets large sparse and dense solving, but substantial overhead is
  incurred for each solve
* Tiny Solver targets small dense solving exclusively, but has zero overhead in
  repeated solves
* Ceres has an incredibly numerically robust LM loop (maybe best in industry)
* Tiny Solver has a "good enough" LM loop for the applications Tiny Solver is
  needed for

See also: Ceres Solver - http://ceres-solver.org/
