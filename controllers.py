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
