import numpy as np
import matplotlib.pyplot as plt

# constants
c = 299792458  # speed of light (m/s)
G = 6.67430e-11  # gravitational constant (m^3 kg^-1 s^-2)
M_earth = 5.972e24  # mass of Earth (kg)
R_earth = 6371e3  # radius of Earth (m)
g = 9.80665  # gravitational acceleration near Earth's surface (m/s^2)

# fractional frequency gradient near Earth
fraction_per_m = g / c**2
fraction_per_cm = fraction_per_m * 0.01
print(f"Fractional frequency gradient per metre: {fraction_per_m}")
print(f"Fractional frequency gradient per centimetre: {fraction_per_cm}")

# height vector for plotting
h = np.linspace(0, 2000, 1000)  # height in metres
frac_shift = g * h / c**2

# compute GPS gravitational and special relativistic corrections
h_orbit = 20200e3  # altitude of GPS satellite (m)
r_ground = R_earth
r_sat = R_earth + h_orbit
phi_ground = -G * M_earth / r_ground
phi_sat = -G * M_earth / r_sat

# gravitational ratio
ratio_gr = np.sqrt(1 + 2 * phi_sat / c**2) / np.sqrt(1 + 2 * phi_ground / c**2)
# special relativity ratio
v = 3.874e3  # orbital velocity (m/s)
ratio_sr = np.sqrt(1 - (v / c)**2)
# combined ratio
ratio_total = ratio_gr * ratio_sr
micro_gr = (ratio_gr - 1) * 86400 * 1e6
micro_sr = (ratio_sr - 1) * 86400 * 1e6
micro_total = (ratio_total - 1) * 86400 * 1e6

print(f"GR correction: {micro_gr:.3f} microseconds per day")
print(f"SR correction: {micro_sr:.3f} microseconds per day")
print(f"Total correction: {micro_total:.3f} microseconds per day")

# plot fractional shift vs height
plt.figure()
plt.plot(h, frac_shift)
plt.xlabel('Height above ground (m)')
plt.ylabel('Fractional frequency shift')
plt.title('Gravitational redshift near Earth (DET/GR)')
plt.grid()
plt.savefig('gravitational_redshift.png')