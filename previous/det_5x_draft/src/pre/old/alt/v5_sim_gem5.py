import math

TAU = 2 * math.pi

def wrap_2pi(theta):
    return theta % TAU

class System:
    def __init__(self, n):
        self.n = n
        self.theta = [0.0] * n
        self.omega = [0.0] * n
        self.K = 1.0  # coupling strength

    def dtheta_at(self, theta_values, i):
        n = self.n
        sum_sin = 0.0
        for j in range(n):
            if j != i:
                sum_sin += math.sin(theta_values[j] - theta_values[i])
        return self.omega[i] + (self.K / n) * sum_sin

    def step_phase(self, dt):
        n = self.n
        theta0 = self.theta.copy()

        # RK2 / midpoint update for stability in strong coupling.
        # IMPORTANT: do NOT wrap at the midpoint; wrapping mid-step introduces discontinuities
        # that can suppress synchronization and distort strong-coupling behavior.
        k1 = [self.dtheta_at(theta0, i) for i in range(n)]
        theta_mid = [theta0[i] + 0.5 * k1[i] for i in range(n)]
        k2 = [self.dtheta_at(theta_mid, i) for i in range(n)]

        self.theta = [wrap_2pi(theta0[i] + dt * k2[i]) for i in range(n)]

def run_sim(s, steps, dt):
    for _ in range(steps):
        s.step_phase(dt)

        # Compute mean phase Ψ
        x = sum(math.cos(th) for th in s.theta)
        y = sum(math.sin(th) for th in s.theta)
        Psi = math.atan2(y, x)

        # Residuals on the circle relative to mean phase Ψ
        resid = [math.atan2(math.sin(th - Psi), math.cos(th - Psi)) for th in s.theta]

        # Additional code can use resid for analysis or convergence checks
        # ...