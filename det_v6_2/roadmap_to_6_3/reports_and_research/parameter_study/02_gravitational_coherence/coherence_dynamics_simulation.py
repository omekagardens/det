import numpy as np
import matplotlib.pyplot as plt

# parameters
alpha_C = 1.0  # growth rate (arbitrary units)
T_d = 17e-3  # experimental decoherence time (s)
lambda_C = 1 / T_d  # decay rate
tau_on = 50e-3  # flux duration (s)
dt = 1e-4  # time step (s)
t_total = 0.1  # total simulation time (s)

# time array
n_steps = int(t_total / dt)
t = np.linspace(0, t_total, n_steps)
C = np.zeros(n_steps)

# simulate
for i in range(1, n_steps):
    J = 1.0 if t[i] < tau_on else 0.0
    dC = alpha_C * J - lambda_C * C[i-1]
    C[i] = C[i-1] + dC * dt

# find decay time after flux stops
ind_start = np.where(t >= tau_on)[0][0]
C0 = C[ind_start]
decay_target = C0 / np.e
ind_decay = np.where(C[ind_start:] <= decay_target)[0][0]
t_decay = t[ind_start + ind_decay] - t[ind_start]
print(f"Estimated decay time: {t_decay:.6f} s")

# plot coherence dynamics
plt.figure()
plt.plot(t * 1000, C)
plt.axvline(tau_on * 1000, color='red', linestyle='--', label='Flux off')
plt.xlabel('Time (ms)')
plt.ylabel('Coherence C (arbitrary units)')
plt.title('DET coherence dynamics: growth and decay')
plt.legend()
plt.grid()
plt.savefig('coherence_dynamics.png')