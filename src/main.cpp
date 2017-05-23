#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];

          /*
          * Calculate steeering angle and throttle using MPC.
          *
          * Both are in between [-1, 1].
          *
          */
          // 1) set steer_value and throttle_value from json data
          double steer_value = j[1]["steering_angle"];
          double throttle_value = j[1]["throttle"];

          // 2) convert to vehicle coordinate systems
          // rotation of psi in negative will convert
          // global coordinate system to vehicle coordinate system
          // converted vehicle_x and vehicle_y in vectors
          // cited:
          // https://gamedev.stackexchange.com/questions/79765/how-do-i-convert-from-the-global-coordinate-space-to-a-local-space

          Eigen::VectorXd vehicle_x(ptsx.size());
          Eigen::VectorXd vehicle_y(ptsy.size());

          for(int i = 0; i < ptsx.size(); i++) {
            double relative_x = ptsx[i] - px;
            double relative_y = ptsy[i] - py;
            vehicle_x[i] = relative_x * cos(-psi) - relative_y * sin(-psi);
            vehicle_y[i] = relative_x * sin(-psi) + relative_y * cos(-psi);
          }

          // 3) fitting 3rd order polynomial
          Eigen::VectorXd coeffs = polyfit(vehicle_x, vehicle_y, 3);
          double cte = polyeval(coeffs, 0);

          // 4) setting up MPC with latency(dt) by modeling dynamic system into vehicle model
          // Orientation error as described in Lesson 18, Lecture 8 "Errors"
          // Current Orientation error (psi0 - psi_desired0) + change in error caused by vehicle moment
          // (v0/Lf * delta0 * dt)
          // psi_desired0 can be calculated as the tangential angle as arctan(f'(xt)) and f' is the derivative
          // of the polynomial
          double epsi = -atan(coeffs[1]);
          double dt = 0.1;
          double Lf = 2.67; // the same Lf values from the class.

          // state (x, y, psi, v) - vehicle x, y, orientation and velocity
          // actuators (delta, a) - steering angle and throttle
          // equations are as follow as in Lesson 18, Lecture 4
          // x[t+1] = x[t] + v[t] * cos(psi[t]) * dt
          // y[t+1] = y[t] + v[t] * sin(psi[t]) * dt
          // psi[t+1] = psi[t] + v[t] / Lf * delat[t] * dt
          // v[t+1] = v[t] + a[t] * dt
          // cte[t+1] = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
          // epsi[t+1] = psi[t] - psi_desired[t] + v[t] * delta[t] / Lf * dt

          double x0 = v * dt;
          double y0 = 0;
          double psi0 = v * steer_value / Lf * dt;
          double v0 = v + throttle_value * dt;
          double cte0 = cte + v * sin(epsi) * dt;
          double epsi0 = epsi + v * steer_value / Lf * dt;

          // fill them in to Eigen vector
          Eigen::VectorXd state(6);
          state << x0, y0, psi0, v0, cte0, epsi0;

          // 5) set the mpc control steering and throttle value
          auto vars = mpc.Solve(state, coeffs);
          steer_value = -vars[0]; // delta
          throttle_value = vars[1]; // a


          json msgJson;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle_value;

          //Display the MPC predicted trajectory
          std::cout << "vars size is: " << vars.size() << endl;
          int count = (vars.size() - 2) / 2; // count is 15 after minus out delta and a
          // initialize the vectors
          vector<double> mpc_x_vals(count);
          vector<double> mpc_y_vals(count);

          // 6) populate the mp x and mpc y values trajectory
          for (int i = 0; i < count; i++) {
            mpc_x_vals[i] = vars[i + 2];
            mpc_y_vals[i] = vars[i + 2 + 1];
          }

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          //Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line

          // 7) initialize the reference line x and y
          for (int i = 0; i < vehicle_x.size(); i++) {
            next_x_vals.push_back(vehicle_x(i));
            next_y_vals.push_back(vehicle_y(i));
          }

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;


          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
