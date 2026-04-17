import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd, output_limits=(-1.0, 1.0)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits
        self.integral = 0
        self.last_error = 0
        self.last_time = None

    def update(self, error, dt):
        if dt <= 0: return 0.0
        
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        
        # Clamp output
        lower, upper = self.output_limits
        output = max(lower, min(upper, output))
        
        self.last_error = error
        return output

class LongitudinalController:
    """Controls acceleration and braking (Forward/Backward)"""
    def __init__(self):
        # PID for maintaining a target TTC of 4.0 seconds
        self.pid = PIDController(Kp=0.2, Ki=0.01, Kd=0.05, output_limits=(-1.0, 0.5))
        self.target_ttc = 4.0

    def calculate(self, ttc, distance, dt):
        # If no threat, stay at cruising speed (slight throttle)
        if ttc >= 99.0: return 0.2 
        
        # Error is the difference between target TTC and current TTC
        # If current TTC < target TTC, error is negative -> negative output (braking)
        error = ttc - self.target_ttc
        
        # Also factor in absolute distance
        if distance < 3.0: return -1.0 # Emergency brake
        
        return self.pid.update(error, dt)

class LateralController:
    """Controls steering angle (Left/Right)"""
    def __init__(self):
        # PID for staying on the center of the path
        self.pid = PIDController(Kp=0.005, Ki=0.0001, Kd=0.001, output_limits=(-1.0, 1.0))

    def calculate(self, current_x, target_x, dt):
        # Error is horizontal pixels from center
        error = target_x - current_x
        return self.pid.update(error, dt)

class PhysicsEngine:
    """Simulates realistic vehicle inertia, friction, and G-Forces"""
    def __init__(self, mass=1500, max_speed=35.0):
        self.mass = mass # kg
        self.max_speed = max_speed # m/s
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.lat_g = 0.0
        self.lon_g = 0.0
        
    def update(self, throttle_brake, steering_input, dt):
        if dt <= 0: return self.current_speed, self.lat_g, self.lon_g
        
        # Longitudinal Dynamics (Acceleration/Braking)
        # throttle_brake is -1.0 to 1.0
        engine_force = throttle_brake * 8000 # Max force in Newtons
        if throttle_brake < 0:
            engine_force = throttle_brake * 15000 # Braking is stronger
            
        # Friction (rolling resistance + aero drag)
        friction = -0.015 * self.mass * 9.81 * (1 if self.current_speed > 0.01 else -1 if self.current_speed < -0.01 else 0)
        drag = -0.4 * 1.2 * (self.current_speed**2) * (1 if self.current_speed > 0 else -1)
        
        total_force = engine_force + friction + drag
        acceleration = total_force / self.mass
        
        # Update speed
        self.current_speed += acceleration * dt
        self.current_speed = max(0, min(self.max_speed, self.current_speed))
        
        # Lateral Dynamics (Steering & G-Force)
        # Smooth steering input to simulate steering rack delay
        self.current_steering += (steering_input - self.current_steering) * min(1.0, dt * 5.0)
        
        # Calculate G-Forces
        self.lon_g = acceleration / 9.81
        
        # Centripetal acceleration: a = v^2 / r. 
        # Assume steering_input relates inversely to turn radius (max steering = tight radius, e.g., 10m)
        turn_radius = 10.0 / (abs(self.current_steering) + 0.001)
        lat_accel = (self.current_speed**2) / turn_radius
        self.lat_g = (lat_accel / 9.81) * (1 if self.current_steering > 0 else -1)
        
        return self.current_speed, self.lat_g, self.lon_g
