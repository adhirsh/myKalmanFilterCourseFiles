import numpy as np
from sim.sim2d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]

options['DRIVE_IN_CIRCLE'] = True
# If False, measurements will be x,y.
# If True, measurements will be x,y, and current angle of the car.
# Required if you want to pass the driving in circle.
options['MEASURE_ANGLE'] = True
options['RECIEVE_INPUTS'] = False

class KalmanFilter:
    def __init__(self):
        # Initial States - all zeros!
        self.x = np.matrix([[0], # x pos
                            [0], # y pos
                            [0], # v
                            [0], # theta
                            [0]])  # theta_dot

        # External Force
        self.u = np.matrix([[0],
                            [0],
                            [0],
                            [0],
                            [0]])

        # Uncertainity Matrix
        self.P = np.matrix([[1000.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1000.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1000.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1000.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1000.0]])

        # Next State Function
        self.F = np.matrix([[1.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 1.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0]])

        # Measurement Function
        self.H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0],  # measure x pos
                            [0.0, 1.0, 0.0, 0.0, 0.0],  # measure y pos
                            [0.0, 0.0, 0.0, 1.0, 0.0]]) # measure theta

        # Measurement Uncertainty
        self.R = np.matrix([[10.0, 0.0, 0.0],
                            [0.0, 10.0, 0.0],
                            [0.0, 0.0, 10.0]])

        # Identity Matrix
        self.I = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0]])
    def predict(self, dt):
        self.F[3,4] = dt
        self.F[0,2] = np.cos(self.x[3])*dt
        self.F[1,2] = np.sin(self.x[3])*dt
        self.x = self.F * self.x + self.u
        self.P = self.F * self.P * np.transpose(self.F)
        return [self.x[0], self.x[1]] #list with x & y position

    def measure_and_update(self,measurements, dt):
        self.F[3,4] = dt
        self.F[0,2] = np.cos(self.x[3])*dt
        self.F[1,2] = np.sin(self.x[3])*dt
        Z = np.matrix(measurements)
        y = np.transpose(Z) - (self.H * self.x)
        S = self.H * self.P * np.transpose(self.H) + self.R
        K = self.P * np.transpose(self.H) * np.linalg.inv(S)
        self.x += K * y
        self.P = (self.I - (K*self.H))*self.P

        #Make it more responsive to changes
        self.P[0,0] += 0.1
        self.P[1,1] += 0.1
        self.P[2,2] += 0.1
        self.P[3,3] += 0.1
        self.P[4,4] += 0.1

        return [self.x[0], self.x[1]]

    def recieve_inputs(self, u_steer, u_pedal):
        return

sim_run(options,KalmanFilter)
