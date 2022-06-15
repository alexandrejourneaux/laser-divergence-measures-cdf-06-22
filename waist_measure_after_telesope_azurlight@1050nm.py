import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

wavelength = 1050.34e-9 # m

def waist(z, w0, z_offset):
    zR = np.pi * w0**2 / wavelength
    return w0 * np.sqrt(1 + ((z - z_offset)/zR)**2)

#%% First config w/ screw issue
z_array = np.array([19.5, 59.5, 87.5, 122]) * 1e-2 
w_array_low = np.array([2538, 2503, 2493, 2447, ]) * 1e-6 / 2
w_array_high = np.array([2746, 2699, 2688, 2634]) * 1e-6 / 2

w_array = np.sqrt(w_array_low * w_array_high)
                   
# popt_low, pcov_low = curve_fit(waist, z_array, w_array_low, p0=[1e-3, 2], bounds=((0, -np.inf), (np.inf, np.inf)))
# popt_high, pcov_high = curve_fit(waist, z_array, w_array_high, p0=[1e-3, 2], bounds=((0, -np.inf), (np.inf, np.inf)))

# popt, pcov = curve_fit(waist, z_array, w_array, p0=[1e-3, 2], bounds=((0, -np.inf), (np.inf, np.inf)))


plt.figure()
# plt.scatter(z_array, w_array_low, c="b", label="My data, w0 = "+f"{popt_low[0]:.2e}")
# plt.scatter(z_array, w_array_high, c="b", label="My data, w0 = "+f"{popt_high[0]:.2e}")
plt.scatter(z_array, w_array, c="b")#, label="My data, w0 = "+f"{popt[0]:.2e}")

z_plot = np.linspace(-1, 3, 300)
# plt.plot(z_plot, [waist(z, *popt_low) for z in z_plot], c="r")
# plt.plot(z_plot, [waist(z, *popt_high) for z in z_plot], c="r")
# plt.plot(z_plot, [waist(z, *popt) for z in z_plot], c="r")

plt.legend()

# print(popt)

#%% l = 139 mm w/out screw issue

z_array = np.array([19.5, 59.5, 87.5, 130, 180]) * 1e-2 
w_array_low = np.array([2592, 2583, 2578, 2583, 2575]) * 1e-6 / 2
w_array_high = np.array([2788, 2763, 2750, 2748, 2721]) * 1e-6 / 2

w_array = np.sqrt(w_array_low * w_array_high)
                   
# popt_low, pcov_low = curve_fit(waist, z_array, w_array_low, p0=[1e-3, 2], bounds=((0, -np.inf), (np.inf, np.inf)))
# popt_high, pcov_high = curve_fit(waist, z_array, w_array_high, p0=[1e-3, 2], bounds=((0, -np.inf), (np.inf, np.inf)))

popt, pcov = curve_fit(waist, z_array, w_array, p0=[1e-3, 2], bounds=((0, -np.inf), (np.inf, np.inf)))


plt.figure()
# plt.scatter(z_array, w_array_low, c="b", label="My data, w0 = "+f"{popt_low[0]:.2e}")
# plt.scatter(z_array, w_array_high, c="b", label="My data, w0 = "+f"{popt_high[0]:.2e}")
plt.scatter(z_array, w_array, c="b")#, label="My data, w0 = "+f"{popt[0]:.2e}")

z_plot = np.linspace(-1, 3, 300)
# plt.plot(z_plot, [waist(z, *popt_low) for z in z_plot], c="r")
# plt.plot(z_plot, [waist(z, *popt_high) for z in z_plot], c="r")
plt.plot(z_plot, [waist(z, *popt) for z in z_plot], c="r")
plt.ylim((-0.0001, 0.00160))
plt.legend()

# print(popt)