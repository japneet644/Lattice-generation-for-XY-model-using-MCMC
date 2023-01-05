import numpy as np
from scipy.optimize import least_squares

class XYModelMetropolisSimulation:
    '''H_matrix is valid only for 2D model'''

    def __init__(self,
                 lattice_shape,
                 beta,
                 J=1,
                 random_state=None):
        self.beta = beta
        self.rs = np.random.RandomState(seed=random_state)
        self.L = self.rs.rand(*lattice_shape)
        self.lattice_shape = lattice_shape
        self.d = len(lattice_shape)
        self.initial_L = self.L.copy()
        self.t = 0
        self.J = J
        self.modified_in_last_step = False

        self.H_matrix = np.zeros(self.L.shape)
        self.H1_matrix = np.zeros(self.L.shape)
        self.H2_matrix = np.zeros(self.L.shape)
        self._calculate_H_matrix()
        self._calculate_H1_matrix()
        self._calculate_H2_matrix()

        self.correlations = []
        for r in range(int(self.L.shape[0] / 2)):
            self.correlations.append(np.cos(self.L[0,0] - self.L[r, 0]))
        self.correlations = np.array(self.correlations).reshape((int(self.L.shape[0] / 2), 1))

        self.H = np.sum(self.H_matrix) / 2
        self.H1= np.sum(self.H1_matrix)/ 2
        self.H2= np.sum(self.H2_matrix)/ 2
        self.H_vals = [self.H]
        self.H1_vals=[self.H1]
        self.H2_vals=[self.H2]

    def make_step(self):
        change_pos = tuple([self.rs.randint(_) for _ in self.lattice_shape])
        new_val = self.rs.rand()
        delta_H = self._get_delta_H(change_pos, new_val)
        delta_H1 = self._get_delta_H1(change_pos, new_val)
        delta_H2 = self._get_delta_H2(change_pos, new_val)
        if (delta_H > 0):
            if (self.rs.rand() < np.exp(-self.beta * delta_H)):
                self._renew_H_matrix(change_pos, new_val)
                self._renew_H1_matrix(change_pos, new_val)
                self._renew_H2_matrix(change_pos, new_val)
                self.L[change_pos] = new_val
                self.H += delta_H
                self.H1 += delta_H1
                self.H2 += delta_H2

                self.modified_in_last_step = True
            else:
                self.modified_in_last_step = False
        else:
            self._renew_H_matrix(change_pos, new_val)
            self._renew_H1_matrix(change_pos, new_val)
            self._renew_H2_matrix(change_pos, new_val)
            self.L[change_pos] = new_val
            self.H += delta_H
            self.H1 += delta_H1
            self.H2 += delta_H2
            self.modified_in_last_step = True
        self.t += 1

    def get_correlations(self):
        return np.mean(self.correlations, axis=1)
    def get_magnetization(self):
        mag_x=np.mean(np.cos(2*np.pi*self.L))
        mag_y=np.mean(np.sin(2*np.pi*self.L))
        return (mag_x**2+mag_y**2)**0.5


    def get_correlation_length(self):
        def optimized_func(x, R, f_log):
            return R + x[0] * f_log - x[0] * x[1]

        A = self.get_correlations()
        bounds = int(len(A) / 5)
        ls = least_squares(optimized_func, [0, 0], kwargs={'R' : np.arange(bounds, len(A) - bounds),
                                                           'f_log' : np.log(np.maximum(A[bounds:-bounds], [1e-10] * (len(A) - 2 * bounds)))})
        return 1 / ls.x[0]

    def get_specific_heat(self):
        actual_vals = int(len(self.H_vals) / 10)
        E_var=np.sum(np.var(self.H_vals[-actual_vals:]))
        return (E_var) * (self.beta ** 2)*(1.0/(self.lattice_shape[0]*self.lattice_shape[1]))

    def get_helicity_modulus(self):
        actual_vals = int(len(self.H_vals) / 10)
        E2=np.mean(self.H2_vals[-actual_vals:])
        E1_sq=np.mean((np.array(self.H1_vals[-actual_vals:]))**2)
        return (1.0/(2*self.lattice_shape[0]*self.lattice_shape[1]))*(E2-self.beta*E1_sq)



    def simulate(self, steps, iters_per_step):
        for i in range(steps):
            for j in range(iters_per_step):
                self.make_step()
            self._compute_space_correlations()
            self.H_vals.append(self.H)
            self.H1_vals.append(self.H1)
            self.H2_vals.append(self.H2)

    def _calculate_H_matrix(self):
        for i in range(self.L.shape[0]):
            for j in range(self.L.shape[1]):
                self.H_matrix[i, j] = 0
                self.H_matrix[i, j] -= np.cos(2 * np.pi * (self.L[i, j] - self.L[i, (j + 1) % self.L.shape[1]]))
                self.H_matrix[i, j] -= np.cos(2 * np.pi * (self.L[i, j] - self.L[i, (j - 1) % self.L.shape[1]]))
                self.H_matrix[i, j] -= np.cos(2 * np.pi * (self.L[i, j] - self.L[(i + 1) % self.L.shape[0], j]))
                self.H_matrix[i, j] -= np.cos(2 * np.pi * (self.L[i, j] - self.L[(i - 1) % self.L.shape[0], j]))
        self.H_matrix *= self.J

    def _calculate_H1_matrix(self):
        for i in range(self.L.shape[0]):
            for j in range(self.L.shape[1]):
                self.H1_matrix[i, j] = 0
                self.H1_matrix[i, j] += np.sin(2 * np.pi * (self.L[i, j] - self.L[i, (j + 1) % self.L.shape[1]]))
                self.H1_matrix[i, j] += np.sin(2 * np.pi * (self.L[i, j] - self.L[i, (j - 1) % self.L.shape[1]]))
                self.H1_matrix[i, j] += np.sin(2 * np.pi * (self.L[i, j] - self.L[(i + 1) % self.L.shape[0], j]))
                self.H1_matrix[i, j] += np.sin(2 * np.pi * (self.L[i, j] - self.L[(i - 1) % self.L.shape[0], j]))
        self.H1_matrix *= self.J

    def _calculate_H2_matrix(self):
        for i in range(self.L.shape[0]):
            for j in range(self.L.shape[1]):
                self.H2_matrix[i, j] = 0
                self.H2_matrix[i, j] += np.cos(2 * np.pi * (self.L[i, j] - self.L[i, (j + 1) % self.L.shape[1]]))
                self.H2_matrix[i, j] += np.cos(2 * np.pi * (self.L[i, j] - self.L[i, (j - 1) % self.L.shape[1]]))
                self.H2_matrix[i, j] += np.cos(2 * np.pi * (self.L[i, j] - self.L[(i + 1) % self.L.shape[0], j]))
                self.H2_matrix[i, j] += np.cos(2 * np.pi * (self.L[i, j] - self.L[(i - 1) % self.L.shape[0], j]))
        self.H2_matrix *= self.J



    def _compute_space_correlations(self):
        correlations = []
        rolled = np.roll(self.L, 1, axis=1)
        for r in range(int(self.L.shape[0] / 2)):
            correlations.append(np.mean(np.cos(2 * np.pi * (self.L - rolled))))
            rolled = np.roll(rolled, 1, axis=1)
        self.correlations = np.concatenate((self.correlations,
                                            np.array(correlations).reshape((len(correlations), 1))),
                                           axis=1)

    def _get_delta_H(self, pos, new_val):
        ans = 0
        old_val = self.L[pos]
        pos_list = list(pos)
        for i in range(len(pos)):
            pos_list[i] += 1
            pos_list[i] %= self.L.shape[i]
            ans += np.cos(2 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                    - np.cos(2 * np.pi * (self.L[tuple(pos_list)] - old_val))
            pos_list[i] -= 2
            pos_list[i] %= self.L.shape[i]
            ans += np.cos(2 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                    - np.cos(2 * np.pi * (self.L[tuple(pos_list)] - old_val))
            pos_list[i] += 1
            pos_list[i] %= self.L.shape[i]
        return -ans * self.J

    def _get_delta_H1(self, pos, new_val):
        ans = 0
        old_val = self.L[pos]
        pos_list = list(pos)
        for i in range(len(pos)):
            pos_list[i] += 1
            pos_list[i] %= self.L.shape[i]
            ans += np.sin(2 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                    - np.sin(2 * np.pi * (self.L[tuple(pos_list)] - old_val))
            pos_list[i] -= 2
            pos_list[i] %= self.L.shape[i]
            ans += np.sin(2 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                    - np.sin(2 * np.pi * (self.L[tuple(pos_list)] - old_val))
            pos_list[i] += 1
            pos_list[i] %= self.L.shape[i]
        return ans * self.J

    def _get_delta_H2(self, pos, new_val):
        ans = 0
        old_val = self.L[pos]
        pos_list = list(pos)
        for i in range(len(pos)):
            pos_list[i] += 1
            pos_list[i] %= self.L.shape[i]
            ans += np.cos(2 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                    - np.cos(2 * np.pi * (self.L[tuple(pos_list)] - old_val))
            pos_list[i] -= 2
            pos_list[i] %= self.L.shape[i]
            ans += np.cos(2 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                    - np.cos(2 * np.pi * (self.L[tuple(pos_list)] - old_val))
            pos_list[i] += 1
            pos_list[i] %= self.L.shape[i]
        return ans * self.J

    def _renew_H_matrix(self, pos, new_val):
        old_val = self.L[pos]
        pos_list = list(pos)
        for i in range(len(pos)):
            pos_list[i] += 1
            pos_list[i] %= self.L.shape[i]
            link_delta_H = np.cos(2 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                            - np.cos(2 * np.pi * (self.L[tuple(pos_list)] - old_val))
            self.H_matrix[tuple(pos_list)] -= link_delta_H * self.J
            self.H_matrix[pos] -= link_delta_H * self.J
            pos_list[i] -= 2
            pos_list[i] %= self.L.shape[i]
            link_delta_H = np.cos(2 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                            - np.cos(2 * np.pi * (self.L[tuple(pos_list)] - old_val))
            self.H_matrix[tuple(pos_list)] -= link_delta_H * self.J
            self.H_matrix[pos] -= link_delta_H * self.J
            pos_list[i] += 1
            pos_list[i] %= self.L.shape[i]
    def _renew_H1_matrix(self, pos, new_val):
        old_val = self.L[pos]
        pos_list = list(pos)
        for i in range(len(pos)):
            pos_list[i] += 1
            pos_list[i] %= self.L.shape[i]
            link_delta_H1 = np.sin(2 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                            - np.sin(2 * np.pi * (self.L[tuple(pos_list)] - old_val))
            self.H1_matrix[tuple(pos_list)] += link_delta_H1 * self.J
            self.H1_matrix[pos] += link_delta_H1 * self.J
            pos_list[i] -= 2
            pos_list[i] %= self.L.shape[i]
            link_delta_H1 = np.sin(2 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                            - np.sin(2 * np.pi * (self.L[tuple(pos_list)] - old_val))
            self.H1_matrix[tuple(pos_list)] += link_delta_H1 * self.J
            self.H1_matrix[pos] += link_delta_H1 * self.J
            pos_list[i] += 1
            pos_list[i] %= self.L.shape[i]
    def _renew_H2_matrix(self, pos, new_val):
        old_val = self.L[pos]
        pos_list = list(pos)
        for i in range(len(pos)):
            pos_list[i] += 1
            pos_list[i] %= self.L.shape[i]
            link_delta_H2 = np.cos(2 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                            - np.cos(2 * np.pi * (self.L[tuple(pos_list)] - old_val))
            self.H2_matrix[tuple(pos_list)] += link_delta_H2 * self.J
            self.H2_matrix[pos] += link_delta_H2 * self.J
            pos_list[i] -= 2
            pos_list[i] %= self.L.shape[i]
            link_delta_H2 = np.cos(2 * np.pi * (self.L[tuple(pos_list)] - new_val)) \
                            - np.cos(2 * np.pi * (self.L[tuple(pos_list)] - old_val))
            self.H2_matrix[tuple(pos_list)] += link_delta_H2 * self.J
            self.H2_matrix[pos] += link_delta_H2 * self.J
            pos_list[i] += 1
            pos_list[i] %= self.L.shape[i]
