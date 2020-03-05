import numpy as np
from sim.sim1d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['CONSTANT_SPEED'] = False

class KalmanFilterToy:
    def __init__(self):
        self.v = 0
        self.prev_x = 0
        self.prev_t = 0
    def predict(self,t):
        dt = t - self.prev_t
        prediction = self.prev_x + self.v*dt
        return prediction #predicted value of x
    def measure_and_update(self,x,t):
        dt = t - self.prev_t
        measured_v = (x - self.prev_x)/dt #update self.v
        self.v += 0.1*(measured_v - self.v) #heigher weight emphasizes current measurement
        self.prev_x = x
        self.prev_t = t
        return


sim_run(options,KalmanFilterToy)
