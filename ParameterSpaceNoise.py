import numpy as np


def policy_distance(a1, a2):
    sq_diff = np.power(a1-a2, 2)
    distance = np.sqrt(np.mean(sq_diff))
    return distance


class AdaptiveParametricNoise(object):
    """
    Referenced from the implementation by Lianming Shi https://github.com/l5shi/Multi-DDPG-with-parameter-noise
    """
    def __init__(self, initial_noise_stdev = 0.2, delta_distance = 0.1,  alpha = 1.01 ):
        self.noise_std_dev = initial_noise_stdev
        self.delta_distance = delta_distance
        self.alpha = alpha
        self.distance = None

    def adapt(self, action, action_perturbed):
        # update value of noise_std_dev
        self.distance = policy_distance(action, action_perturbed)
        if self.distance < self.delta_distance:
            self.noise_std_dev *= self.alpha
        else:
            self.noise_std_dev /= self.alpha
        

    

        