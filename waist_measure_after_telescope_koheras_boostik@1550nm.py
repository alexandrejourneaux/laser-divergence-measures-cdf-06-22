import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tikzplotlib import clean_figure as tikz_clean, save as tikz_save

def dx(N):
    res = np.diag(np.ones((N-1,), dtype=float), k=1) - np.diag(np.ones((N-1,), dtype=float), k=-1)
    res[0, :3] = [-3, 4, -1]
    res[-1, -3:] = [1, -4, 3]
    
    return res

def gaussian(x, amp, avg, w):
    return amp * np.exp(- 2*(x - avg)**2 / w**2)

#%% d = 15 cm
# z15 = np.arange(2.5, 6.5, 0.1)
# power15 = np.array([5.70, 5.70, 5.68, 5.65, 5.63, 5.60, 5.58, 5.52, 5.51, 5.5, 
#                   5.42, 5.33, 5.26, 5.15, 5.03, 4.86, 4.67, 4.41, 4.05, 3.63, 
#                   3.2, 2.7, 2.25, 1.86, 1.5, 1.16, 0.86, 0.65, 0.48, 0.33, 
#                   0.23, 0.15, 0.10, 0.07, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01])

# beam_shape15 = - dx(len(power15))@power15

# popt15, pcov15 = curve_fit(gaussian, z15, beam_shape15, p0=[1, 5, 1], bounds=([-np.inf, -np.inf, 0],[np.inf, np.inf, np.inf]))

# plt.figure()
# plt.scatter(z15, beam_shape15)
# plt.plot(z15, gaussian(z15, *popt15), c="r")



#%% d = 30 cm

# z30 = np.arange(1.0, 8, 0.1)
# power30 = np.array([4.97, 4.97, 4.98, 4.98, 4.97, 4.95, 4.94, 4.94, 4.93, 4.94,
#                   4.95, 4.94, 4.94, 4.95, 4.96, 4.96, 4.93, 4.92, 4.93, 4.96,
#                   4.97, 4.90, 4.80, 4.67, 4.65, 4.58, 4.56, 4.51, 4.44, 4.35, 4.26, 3.98, 3.81, 3.59, 3.33, 3.03, 2.70,
#                   2.32, 1.9, 1.56, 1.24, 0.94, 0.70, 0.49, 0.36, 0.25, 0.17, 0.11, 0.072, 0.047, 0.031, 0.020, 0.014, 0.011, 0.0088, 0.0074, 0.0064, 0.0056, 0.0052, 0.0049, 0.0040, 0.0044, 0.0034, 0.0033, 0.0036, 0.0036, 0.0036, 0.0035, 0.0033, 0.0033])

# beam_shape30 = - dx(len(power30))@power30

# popt30, pcov30 = curve_fit(gaussian, z30, beam_shape30, p0=[1, 5, 1], bounds=([-np.inf, -np.inf, 0],[np.inf, np.inf, np.inf]))

# plt.figure()
# plt.scatter(z30, beam_shape30)
# plt.plot(z30, gaussian(z30, *popt30), c="r")

#%% d = 93 cm

z93 = np.arange(0, 10.1, 0.1)
power93 = np.array([4.34, 4.34, 4.34, 4.34, 4.34, 4.35, 4.34, 4.35, 4.35, 4.35,
                    4.34, 4.35, 4.34, 4.35, 4.34, 4.34, 4.34, 4.34, 4.34, 4.34,
                    4.34, 4.34, 4.33, 4.33, 4.32, 4.32, 4.32, 4.32, 4.31,
                    4.31, 4.30, 4.29, 4.26, 4.24, 4.21, 4.19, 4.14, 4.09,
                    4.02, 3.94, 3.87, 3.79, 3.68, 3.58, 3.46, 3.32, 3.18, 3.03, 
                    2.87, 2.71, 2.54, 2.37, 2.19, 2.02, 1.85, 1.68, 1.5, 1.35,
                    1.19, 1.04, 0.906, 0.777, 0.663, 0.560, 0.468, 0.384,
                    0.311, 0.25, 0.198, 0.155, 0.118, 0.089, 0.066, 0.0493,
                    0.0368, 0.0278, 0.0214, 0.0168, 0.0137, 0.0112, 0.0085,
                    0.0078, 0.0067, 0.00605, 0.00520, 0.00497, 0.00462,
                    0.00440, 0.00412, 0.00408, 0.00380, 0.00370, 0.00350,
                    0.00363, 0.00336, 0.00344, 0.00320, 0.00316, 0.00320,
                    0.00325, 0.00318])

beam_shape93 = - dx(len(power93))@power93

popt93, pcov93 = curve_fit(gaussian, z93, beam_shape93, p0=[1, 5, 1], bounds=([-np.inf, -np.inf, 0],[np.inf, np.inf, np.inf]))

plt.figure()
plt.scatter(z93, beam_shape93)
plt.plot(z93, gaussian(z93, *popt93), c="r")
print(popt93)

#%% d = 27.5 cm

z27 = np.arange(1, 11.2, 0.2)
power27 = np.array([3.62, 3.63, 3.64, 3.63, 3.61, 3.61, 3.63, 3.62, 3.63, 3.65, 3.66, 3.67, 3.67, 3.68, 3.69, 3.69, 3.71, 3.71, 3.69, 3.68, 3.66, 3.63, 3.57, 3.48, 3.38, 3.21, 2.99, 2.73, 2.43, 2.09, 1.75, 1.43, 1.12, 0.833, 0.594, 0.401, 0.254, 0.154, 0.0872, 0.0490, 0.0282, 0.0162, 0.00882, 0.00544, 0.00415, 0.00350, 0.00320, 0.00298, 0.00296, 0.00294, 0.00285])

beam_shape27 = - dx(len(power27))@power27

popt27, pcov27 = curve_fit(gaussian, z27, beam_shape27, p0=[1, 5, 1], bounds=([-np.inf, -np.inf, 0],[np.inf, np.inf, np.inf]))

plt.figure()
plt.scatter(z27, beam_shape27)
plt.plot(z27, gaussian(z27, *popt27), c="r")
print(popt27)



#%% Final interpolation

wavelength = 1550.12e-9 # m

def waist(z, w0, z_offset):
    zR = np.pi * w0**2 / wavelength
    return w0 * np.sqrt(1 + ((z - z_offset)/zR)**2)

z_array = np.array([27.5, 93])*1e-2
w_array = np.array([popt27[2], popt93[2]])*1e-3
                   
popt, pcov = curve_fit(waist, z_array, w_array, bounds=((0, -np.inf), (np.inf, np.inf)))

plt.figure()
plt.scatter(z_array, w_array*1000)

z_plot = np.linspace(-3, 3, 400)
plt.plot(z_plot, [waist(z, *popt)*1000 for z in z_plot], c="r", label=f"w0 = {round(1000*popt[0]*1e3)/1000} mm, z0 = {round(1000*popt[1]*1e3)/1000} mm")
plt.xlabel("Distance to telescope (m)")
plt.ylabel("Beam waist (mm)")
plt.title("Beam divergence")
plt.legend()
plt.savefig("koheras_beam_after_telescope.png")

print(popt[0] * 1e3)