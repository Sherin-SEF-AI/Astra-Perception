#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>

namespace py = pybind11;

class PIDController {
public:
    PIDController(double Kp, double Ki, double Kd, std::pair<double, double> output_limits)
        : Kp(Kp), Ki(Ki), Kd(Kd), output_limits(output_limits), integral(0.0), last_error(0.0) {}

    double update(double error, double dt) {
        if (dt <= 0) return 0.0;
        
        integral += error * dt;
        double derivative = (error - last_error) / dt;
        
        double output = (Kp * error) + (Ki * integral) + (Kd * derivative);
        
        // Clamp output
        output = std::max(output_limits.first, std::min(output_limits.second, output));
        
        last_error = error;
        return output;
    }

private:
    double Kp, Ki, Kd;
    std::pair<double, double> output_limits;
    double integral;
    double last_error;
};

class LongitudinalController {
public:
    LongitudinalController() 
        : pid(0.2, 0.01, 0.05, {-1.0, 0.5}), target_ttc(4.0) {}

    double calculate(double ttc, double distance, double dt) {
        if (ttc >= 99.0) return 0.2;
        
        double error = ttc - target_ttc;
        
        if (distance < 3.0) return -1.0;
        
        return pid.update(error, dt);
    }

private:
    PIDController pid;
    double target_ttc;
};

class LateralController {
public:
    LateralController()
        : pid(0.005, 0.0001, 0.001, {-1.0, 1.0}) {}

    double calculate(double current_x, double target_x, double dt) {
        double error = target_x - current_x;
        return pid.update(error, dt);
    }

private:
    PIDController pid;
};

class PhysicsEngine {
public:
    PhysicsEngine(double mass = 1500.0, double max_speed = 35.0)
        : mass(mass), max_speed(max_speed), current_speed(0.0), current_steering(0.0), lat_g(0.0), lon_g(0.0) {}

    std::tuple<double, double, double> update(double throttle_brake, double steering_input, double dt) {
        if (dt <= 0) return {current_speed, lat_g, lon_g};
        
        // Longitudinal Dynamics
        double engine_force = throttle_brake * (throttle_brake < 0 ? 15000.0 : 8000.0);
        
        double friction = -0.015 * mass * 9.81 * (current_speed > 0.01 ? 1.0 : (current_speed < -0.01 ? -1.0 : 0.0));
        double drag = -0.4 * 1.2 * (current_speed * current_speed) * (current_speed > 0 ? 1.0 : -1.0);
        
        double total_force = engine_force + friction + drag;
        double acceleration = total_force / mass;
        
        current_speed += acceleration * dt;
        current_speed = std::max(0.0, std::min(max_speed, current_speed));
        
        // Lateral Dynamics
        current_steering += (steering_input - current_steering) * std::min(1.0, dt * 5.0);
        
        lon_g = acceleration / 9.81;
        
        double turn_radius = 10.0 / (std::abs(current_steering) + 0.001);
        double lat_accel = (current_speed * current_speed) / turn_radius;
        lat_g = (lat_accel / 9.81) * (current_steering > 0 ? 1.0 : -1.0);
        
        return {current_speed, lat_g, lon_g};
    }

private:
    double mass, max_speed;
    double current_speed, current_steering, lat_g, lon_g;
};

PYBIND11_MODULE(astra_core, m) {
    py::class_<PIDController>(m, "PIDController")
        .def(py::init<double, double, double, std::pair<double, double>>())
        .def("update", &PIDController::update);

    py::class_<LongitudinalController>(m, "LongitudinalController")
        .def(py::init<>())
        .def("calculate", &LongitudinalController::calculate);

    py::class_<LateralController>(m, "LateralController")
        .def(py::init<>())
        .def("calculate", &LateralController::calculate);

    py::class_<PhysicsEngine>(m, "PhysicsEngine")
        .def(py::init<double, double>(), py::arg("mass") = 1500.0, py::arg("max_speed") = 35.0)
        .def("update", &PhysicsEngine::update);
}
