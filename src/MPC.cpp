#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// 1) As I recalled from Lesson 19 Lecture 5 "Length and Duration":
// A good approach to setting N, dt and T is to first decide T, then tune dt and N appropriately.
// T = N * dt, and I started with T to be between 1.5 ~ 2.5 second
// it seems the larger dt the less frequent actuations happen so it is harder to predict
// reference trajectory.
// So I started dt between 200 ms to 100 ms and then finally set dt to be 100 ms. So N is between
// 15 and 25. After tuning those variables, I settled with N = 15 and dt = 0.1
size_t N = 15;
double dt = 0.1;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// 2) setting reference cte, epsi and reference velocity as learned from Lesson 18, Lecture 8 "Errors"
// to start with cte as 0, epsi as 0 and velocity as 50mph
const double ref_cte = 0;
const double ref_epsi = 0;
const double ref_v = 50;

// 3) for the solver as it takes state variables and actuators in a vector,
// try to line up all the variables in the following orders: x, y, psi, v, cte, epsi, delta, a
const size_t x_start = 0;
const size_t y_start = x_start + N;
const size_t psi_start = y_start + N;
const size_t v_start = psi_start + N;
const size_t cte_start = v_start + N;
const size_t epsi_start = cte_start + N;
const size_t delta_start = epsi_start + N;
const size_t a_start = delta_start + N - 1;


class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;

  FG_eval(Eigen::VectorXd coeffs) {
    this->coeffs = coeffs;
  }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;

  void operator()(ADvector& fg, const ADvector& vars) {
    // 4) Implementation of MPC
    // fg a vector of constraints, x is a vector of constraints.
    // the cost is stored in the first element of fg (fg[0])
    fg[0] = 0;

    // the part of the cost based on the reference state
    // ======= COSTS ===================================
    // 5) cost multipliers are tunned here to keep solver to have sequential values closer
    for (int i = 0; i < N; i++){
      //fg[0] += 5000 * CppAD::pow(vars[cte_start + i] - ref_cte, 2);
      //fg[0] += 5000 * CppAD::pow(vars[epsi_start + i] - ref_epsi, 2);
      //fg[0] += 10 * CppAD::pow(vars[v_start + 1] - ref_v, 2);
      fg[0] += CppAD::pow(vars[cte_start + i] - ref_cte, 2);
      fg[0] += CppAD::pow(vars[epsi_start + i] - ref_epsi, 2);
      fg[0] += CppAD::pow(vars[v_start + 1] - ref_v, 2);
    }

    // Minimize the use of actuators
    for (int i = 0; i < N - 1; i++) {
      //fg[0] += 10 * CppAD::pow(vars[delta_start + i], 2);
      //fg[0] += 50 * CppAD::pow(vars[a_start + i], 2);
      fg[0] += 50 * CppAD::pow(vars[delta_start + i], 2);
      fg[0] += 60 * CppAD::pow(vars[a_start + i], 2);
    }

    // Minimize the value gap between sequential actuations.
    for (int i = 0; i < N -2; i++) {
      //fg[0] += 1000 * CppAD::pow(vars[delta_start + i + 1] - vars[delta_start + i], 2);
      //fg[0] += 100 * CppAD::pow(vars[a_start + i + 1] - vars[a_start + i], 2);
      fg[0] += CppAD::pow(vars[delta_start + i + 1] - vars[delta_start + i], 2);
      fg[0] += CppAD::pow(vars[a_start + i + 1] - vars[a_start + i], 2);
    }

    // ===== CONSTRAINTS ============================
    // 6) Setting up the constraints.
    // since cost are in fg[0], fg[1] is for constraints
    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    // the rest of the constraints
    for (int i = 0; i < N - 1; i++) {
      // The state at time t+1
      AD<double> x1 = vars[x_start + i + 1];
      AD<double> y1 = vars[y_start + i + 1];
      AD<double> psi1 = vars[psi_start + i + 1];
      AD<double> v1 = vars[v_start + i + 1];
      AD<double> cte1 = vars[cte_start + i + 1];
      AD<double> epsi1 = vars[epsi_start + i + 1];

      // The state at time t
      AD<double> x0 = vars[x_start + i];
      AD<double> y0 = vars[y_start + i];
      AD<double> psi0 = vars[psi_start + i];
      AD<double> v0 = vars[v_start + i];
      AD<double> cte0 = vars[cte_start + i];
      AD<double> epsi0 = vars[epsi_start + i];

      // Only consider the actuation at time t
      AD<double> delta0 = vars[delta_start + i];
      AD<double> a0 = vars[a_start + i];

      // f(x) = c0 + c1 * x0 + c2 * x02 + c3 * x03 (third order)
      AD<double> f0 = coeffs[0] + coeffs[1] * x0 + coeffs[2] * CppAD::pow(x0, 2) + coeffs[3] * CppAD::pow(x0, 3);
      AD<double> psides0 = CppAD::atan(coeffs[1] + 2 * coeffs[2] * x0 + 3 * coeffs[3] * CppAD::pow(x0, 2));

      // Here's `x` to get you started.
      // The idea here is to constraint this value to be 0.
      //
      // Recall the equations for the model:
      // x_[t+1] = x[t] + v[t] * cos(psi[t]) * dt
      // y_[t+1] = y[t] + v[t] * sin(psi[t]) * dt
      // psi_[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
      // v_[t+1] = v[t] + a[t] * dt
      // cte[t+1] = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
      // epsi[t+1] = psi[t] - psides[t] + v[t] * delta[t] / Lf * dt
      fg[2 + x_start + i] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[2 + y_start + i] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[2 + psi_start + i] = psi1 - (psi0 + v0 * delta0 / Lf * dt);
      fg[2 + v_start + i] = v1 - (v0 + a0 * dt);
      fg[2 + cte_start + i] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
      fg[2 + epsi_start + i] = epsi1 - ((psi0 - psides0) + v0 * delta0 / Lf * dt);
    }

  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  //size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  // Set the number of model variables (includes both states and inputs).
  // For example: If the state is a 4 element vector, the actuators is a 2
  // element vector and there are 10 timesteps. The number of variables is:
  //
  // 4 * 10 + 2 * 9
  // 1) numbers of state * N + (N -1) * actuators = 4 * N + 2 * (N-1)
  double x = state[0];
  double y = state[1];
  double psi = state[2];
  double v = state[3];
  double cte = state[4];
  double epsi = state[5];

  size_t n_vars = 6 * N + 2 * (N - 1);
  // Set the number of constraints
  size_t n_constraints = N * 6;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }

  // 2) Set the initial values
  vars[x_start] = x;
  vars[y_start] = y;
  vars[psi_start] = psi;
  vars[v_start] = v;
  vars[cte_start] = cte;
  vars[epsi_start] = epsi;


  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);

  // 3) Set lower and upper limits for variables.
  for (int i = 0; i < delta_start; i++){
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] = 1.0e19;
  }

  // The upper and lower limits of delta are set to -20 and 20
  // degrees (values in radians).
  // NOTE: Feel free to change this to something else.
  for (int i = delta_start; i < a_start; i++) {
    vars_lowerbound[i] = -0.349066;
    vars_upperbound[i] = 0.349066;
  }

  // Acceleration/decceleration upper and lower limits.
  // NOTE: Feel free to change this to something else.
  for (int i = a_start; i < n_vars; i++) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] = 1.0;
  }

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (int i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

  constraints_lowerbound[x_start] = x;
  constraints_lowerbound[y_start] = y;
  constraints_lowerbound[psi_start] = psi;
  constraints_lowerbound[v_start] = v;
  constraints_lowerbound[cte_start] = cte;
  constraints_lowerbound[epsi_start] = epsi;

  constraints_upperbound[x_start] = x;
  constraints_upperbound[y_start] = y;
  constraints_upperbound[psi_start] = psi;
  constraints_upperbound[v_start] = v;
  constraints_upperbound[cte_start] = cte;
  constraints_upperbound[epsi_start] = epsi;

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  // Return the first actuator values. The variables can be accessed with
  // `solution.x[i]`.
  //
  // {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
  // creates a 2 element double vector.
  std:vector<double> results;
  double delta = 0;
  double a = 0.0;

  int n_mean = 3;
  for (int i = 0; i < n_mean; i++){
    delta += solution.x[delta_start + i] / n_mean;
    a += solution.x[a_start + i] /n_mean;
  }

  results.push_back(delta);
  results.push_back(a);

  for (int i = 0; i < N; ++i) {
    results.push_back(solution.x[x_start + i]);
    results.push_back(solution.x[y_start + i]);
  }
  return results;
  //return {solution.x[delta_start], solution.x[a_start]};
}
