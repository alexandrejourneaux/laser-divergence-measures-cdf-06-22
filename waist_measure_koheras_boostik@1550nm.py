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

#%% d = 45 cm

z45 = np.arange(1.0, 8, 0.1)
power45 = np.array([4.96, 4.96, 4.96, 4.96, 4.96, 4.96, 4.97, 4.97, 4.97, 4.97, 
                    4.97, 4.97, 4.97, 4.97, 4.96, 4.96, 4.96, 4.96, 4.95, 4.93,
                    4.91, 4.87, 4.82, 4.72, 4.59, 4.41, 4.19, 3.92, 3.59, 3.22, 
                    2.84, 2.43, 2.05, 1.68, 1.33, 1.02, 0.768, 0.563, 0.392, 
                    0.268, 0.178, 0.114, 0.072, 0.046, 0.029, 0.019, 0.014, 
                    0.010, 0.008, 0.0071, 0.0063, 0.0058, 0.0054, 0.0051, 
                    0.0048, 0.0046, 0.0044, 0.0042, 0.0041, 0.0039, 0.0038,
                    0.0036, 0.0035, 0.0033, 0.0032, 0.0030, 0.0029, 0.0029, 0.0029, 0.0028])

beam_shape45 = - dx(len(power45))@power45

popt45, pcov45 = curve_fit(gaussian, z45, beam_shape45, p0=[1, 5, 1], bounds=([-np.inf, -np.inf, 0],[np.inf, np.inf, np.inf]))

plt.figure()
plt.scatter(z45, beam_shape45/np.max(beam_shape45))#, label="Mesures")
plt.plot(z45, gaussian(z45, *popt45)/np.max(beam_shape45), c="r")#, label=f"Interpolation")
plt.xlabel("Position de la lame (mm)")
plt.ylabel("Dérivée de la puissance (u.a.)")
plt.legend()
#tikz_clean()
#tikz_save("interpolation_gaussienne_1D.tikz")

#%% d = 60 cm

z60 = np.arange(1.0, 8, 0.1)
power60 = np.array([5.68, 5.68, 5.66, 5.67, 5.68, 5.66, 5.66, 5.67, 5.66, 5.67,
                    5.67, 5.68, 5.67, 5.68, 5.69, 5.68, 5.66, 5.67, 5.67, 5.68, 
                    5.66, 5.67, 5.64, 5.63, 5.62, 5.56, 5.47, 5.33, 5.18, 4.97,
                    4.69, 4.38, 4.02, 3.60, 3.18, 2.71, 2.27, 1.84, 1.45, 1.11,
                    0.827, 0.595, 0.411, 0.279, 0.184, 0.116, 0.072, 0.047,
                    0.030, 0.020, 0.0146, 0.0111, 0.0088, 0.0077, 0.0069,
                    0.0063, 0.0060, 0.0057, 0.0054, 0.0050, 0.0048, 0.0046,
                    0.0043, 0.0042, 0.0037, 0.0035, 0.0033, 0.0030, 0.0028, 
                    0.0026])

beam_shape60 = - dx(len(power60))@power60

popt60, pcov60 = curve_fit(gaussian, z60, beam_shape60, p0=[1, 5, 1], bounds=([-np.inf, -np.inf, 0],[np.inf, np.inf, np.inf]))

plt.figure()
plt.scatter(z60, beam_shape60)
plt.plot(z60, gaussian(z60, *popt60), c="r")

#%% d = 90 cm

z90 = np.arange(1.0, 8, 0.1)
power90 = np.array([5.93, 5.92, 5.90, 5.90, 5.90, 5.90, 5.90, 5.91, 5.91, 5.90,
                    5.91, 5.92, 5.93, 5.91, 5.92, 5.93, 5.93, 5.93, 5.92, 5.91,
                    5.89, 5.87, 5.85, 5.78, 5.69, 5.55, 5.35, 5.12, 4.82, 4.48,
                    4.05, 3.63, 3.18, 2.74, 2.30, 1.88, 1.49, 1.13, 0.839,
                    0.609, 0.423, 0.282, 0.179, 0.115, 0.072, 0.043, 0.025,
                    0.015, 0.011, 0.0086, 0.0074, 0.0064, 0.0058, 0.0053,
                    0.0044, 0.0042, 0.0040, 0.0039, 0.0037, 0.0036, 0.0035,
                    0.0034, 0.0032, 0.0032, 0.0031, 0.0030, 0.0030, 0.0029,
                    0.0029, 0.0029])

beam_shape90 = - dx(len(power90))@power90

popt90, pcov90 = curve_fit(gaussian, z90, beam_shape90, p0=[1, 5, 1], bounds=([-np.inf, -np.inf, 0],[np.inf, np.inf, np.inf]))

plt.figure()
plt.scatter(z90, beam_shape90)
plt.plot(z90, gaussian(z90, *popt90), c="r")

#%% d = 140 cm

z140 = np.arange(1.0, 8, 0.1)
power140 = np.array([6.55, 6.54, 6.54, 6.54, 6.54, 6.54, 6.54, 6.54, 6.54, 
                     6.54, 6.54, 6.54, 6.54, 6.54, 6.55, 6.55, 6.54, 6.54, 
                     6.54, 6.54, 6.54, 6.54, 6.54, 6.54, 6.54, 6.54, 6.53, 
                     6.52, 6.50, 6.46, 6.42, 6.35, 6.27, 6.14, 5.99, 5.78, 
                     5.53, 5.23, 4.88, 4.49, 4.05, 3.57, 3.12, 2.67, 2.24, 
                     1.83, 1.45, 1.14, 0.868, 0.650, 0.460, 0.314, 0.211, 
                     0.137, 0.0871, 0.0516, 0.0295, 0.0172, 0.0103, 0.0069, 
                     0.0063, 0.0056, 0.0054, 0.0051, 0.0051, 0.0050, 0.0049, 
                     0.0048, 0.0046, 0.0044])

beam_shape140 = - dx(len(power140))@power140

popt140, pcov140 = curve_fit(gaussian, z140, beam_shape140, p0=[1, 5, 1], bounds=([-np.inf, -np.inf, 0],[np.inf, np.inf, np.inf]))

plt.figure()
plt.scatter(z140, beam_shape140)
plt.plot(z140, gaussian(z140, *popt140), c="r")

#%% d = 190 cm

z190 = np.arange(1.0, 9.1, 0.1)
power190 = np.array([6.41, 6.41, 6.40, 6.40, 6.39, 6.39, 6.39, 6.39, 6.39, 6.38, 6.38, 6.37, 6.37, 6.37, 6.35, 6.35, 6.35, 6.36, 6.36, 6.36, 6.35, 6.35, 6.34, 6.33, 6.32, 6.31, 6.30, 6.28, 6.25, 6.21, 6.15, 6.09, 6.01, 5.88, 5.74, 5.56, 5.33, 5.07, 4.76, 4.43, 4.06, 3.68, 3.27, 2.86, 2.48, 2.07, 1.70, 1.40, 1.12, 0.883, 0.675, 0.509, 0.372, 0.268, 0.189, 0.127, 0.082, 0.053, 0.035, 0.030, 0.0154, 0.0111, 0.0083, 0.0067, 0.0058, 0.0055, 0.0048, 0.0046, 0.0044, 0.0044, 0.0041, 0.0040, 0.0038, 0.0037, 0.0036, 0.0036, 0.0034, 0.0034, 0.0033, 0.0034, 0.0033])

beam_shape190 = - dx(len(power190))@power190

popt190, pcov190 = curve_fit(gaussian, z190, beam_shape190, p0=[1, 5, 1], bounds=([-np.inf, -np.inf, 0],[np.inf, np.inf, np.inf]))

plt.figure()
plt.scatter(z190, beam_shape190)
plt.plot(z190, gaussian(z190, *popt190), c="r")

#%% d = 250 cm

z250 = np.arange(1.0, 9.4, 0.1)
power250 = np.array([5.14, 5.12, 5.12, 5.14, 5.13, 5.12, 5.14, 5.14, 5.10, 5.10, 5.11, 5.11, 5.09, 5.07, 5.06, 5.04, 4.99, 5.00, 5.00, 4.97, 4.95, 4.93, 4.89, 4.87, 4.82, 4.75, 4.69, 4.61, 4.47, 4.33, 4.18, 4.01, 3.83, 3.60, 3.37, 3.09, 2.82, 2.56, 2.28, 2.00, 1.75, 1.49, 1.27, 1.06, 0.888, 0.694, 0.552, 0.437, 0.335, 0.258, 0.192, 0.142, 0.106, 0.081, 0.058, 0.042, 0.031, 0.022, 0.016, 0.012, 0.0098, 0.0081, 0.0067, 0.0057, 0.0051, 0.0047, 0.0042, 0.0039, 0.0036, 0.0034, 0.0032, 0.0031, 0.0030, 0.0029, 0.0028, 0.0027, 0.0027, 0.0027, 0.0027, 0.0027, 0.0027, 0.0027, 0.0027, 0.0027])

beam_shape250 = - dx(len(power250))@power250

popt250, pcov250 = curve_fit(gaussian, z250, beam_shape250, p0=[1, 5, 1], bounds=([-np.inf, -np.inf, 0],[np.inf, np.inf, np.inf]))

plt.figure()
plt.scatter(z250, beam_shape250)
plt.plot(z250, gaussian(z250, *popt250), c="r")

#%% Final interpolation

wavelength = 1550.12e-9 # m

def waist(z, w0, z_offset):
    zR = np.pi * w0**2 / wavelength
    return w0 * np.sqrt(1 + ((z - z_offset)/zR)**2)

z_array = np.array([45, 60, 90, 140, 190, 250])*1e-2
w_array = np.array([popt45[2], popt60[2], popt90[2], popt140[2], popt190[2], popt250[2]])*1e-3
                   
popt, pcov = curve_fit(waist, z_array, w_array, bounds=((0, -np.inf), (np.inf, np.inf)))

plt.figure(figsize=(6, 2))
plt.scatter(z_array, w_array*1000)

z_plot = np.linspace(0, 3, 400)
plt.plot(z_plot, [waist(z, *popt)*1000 for z in z_plot], c="r")#, label="Interpolation") #f"w0 = {round(1000*popt[0]*1e3)/1000} mm, z0 = {round(1000*popt[1]*1e3)/1000} mm")
plt.xlabel("Distance au collimateur (m)")
plt.ylabel("Demi-largeur à $1/e^2$ du faisceau (mm)")
#plt.savefig("koheras_beam_divergence.png")
#tikz_clean()
#tikz_save("beam_divergence_1550.tikz")


print("w0 = ", popt[0]*1e3, " mm")