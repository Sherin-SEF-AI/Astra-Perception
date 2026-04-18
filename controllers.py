import numpy as np
import math

try:
    from astra_core import PIDController, LongitudinalController, LateralController, PhysicsEngine
    print("Astra: Using native C++ acceleration for Controllers")
except ImportError:
    print("Astra: Native C++ extensions not found, using Python fallbacks")

    class PIDController:
        def __init__(self, Kp, Ki, Kd, output_limits=(-1.0, 1.0)):
            self.Kp = Kp
            self.Ki = Ki
            self.Kd = Kd
            self.output_limits = output_limits
            self.integral = 0
            self.last_error = 0

        def update(self, error, dt):
            if dt <= 0: return 0.0
            self.integral += error * dt
            derivative = (error - self.last_error) / dt
            output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
            lower, upper = self.output_limits
            output = max(lower, min(upper, output))
            self.last_error = error
            return output

    class LongitudinalController:
        def __init__(self):
            self.pid = PIDController(Kp=0.2, Ki=0.01, Kd=0.05, output_limits=(-1.0, 0.5))
            self.target_ttc = 4.0
        def calculate(self, ttc, distance, dt):
            if math.isnan(ttc) or math.isnan(distance): return -1.0
            if ttc >= 99.0: return 0.2 
            error = ttc - self.target_ttc
            if distance < 3.0: return -1.0
            return self.pid.update(error, dt)

    class LateralController:
        def __init__(self):
            self.pid = PIDController(Kp=0.005, Ki=0.0001, Kd=0.001, output_limits=(-1.0, 1.0))
        def calculate(self, current_x, target_x, dt):
            if math.isnan(current_x) or math.isnan(target_x): return 0.0
            error = target_x - current_x
            return self.pid.update(error, dt)

    class PhysicsEngine:
        def __init__(self, mass=1500, max_speed=35.0):
            self.mass = mass
            self.max_speed = max_speed
            self.current_speed = 0.0
            self.current_steering = 0.0
            self.lat_g = 0.0
            self.lon_g = 0.0
        def update(self, throttle_brake, steering_input, dt):
            if dt <= 0: return self.current_speed, self.lat_g, self.lon_g
            engine_force = throttle_brake * (8000.0 if throttle_brake >= 0 else 15000.0)
            friction = -0.015 * self.mass * 9.81 * (1.0 if self.current_speed > 0.01 else -1.0 if self.current_speed < -0.01 else 0.0)
            drag = -0.4 * 1.2 * (self.current_speed**2) * (1.0 if self.current_speed > 0 else -1.0)
            total_force = engine_force + friction + drag
            acceleration = total_force / self.mass
            self.current_speed += acceleration * dt
            self.current_speed = max(0.0, min(self.max_speed, self.current_speed))
            self.current_steering += (steering_input - self.current_steering) * min(1.0, dt * 5.0)
            self.lon_g = acceleration / 9.81
            steer_abs = abs(self.current_steering)
            if steer_abs > 0.001:
                turn_radius = 10.0 / steer_abs
                lat_accel = (self.current_speed**2) / turn_radius
                self.lat_g = (lat_accel / 9.81) * (1.0 if self.current_steering > 0 else -1.0)
            else:
                self.lat_g = 0.0
            return self.current_speed, self.lat_g, self.lon_g
