import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

wavelength = 1050.34e-9 # m

def waist(z, w0, z_offset):
    zR = np.pi * w0**2 / wavelength
    return w0 * np.sqrt(1 + ((z - z_offset)/zR)**2)


z_array = np.array([19.5, 59.5, 87.5, 130, 180]) * 1e-2 
w_array_low = np.array([2592, 2583, 2578, 2583, 2575]) * 1e-6 / 2
w_array_high = np.array([2788, 2763, 2750, 2748, 2721]) * 1e-6 / 2

w_array = np.sqrt(w_array_low * w_array_high)

popt, pcov = curve_fit(waist, z_array, w_array, p0=[1e-3, 2], bounds=((0, -np.inf), (np.inf, np.inf)))


plt.figure()
plt.scatter(z_array, w_array*1000, c="b")

z_plot = np.linspace(-1, 3, 400)
plt.plot(z_plot, [waist(z, *popt)*1000 for z in z_plot], c="r", label=f"w0 = {round(1000*popt[0]*1e3)/1000} mm, z0 = {round(1000*popt[1]*1e3)/1000} mm")
plt.xlabel("Distance to telescope (m)")
plt.ylabel("Beam waist (mm)")
plt.title("Beam divergence")
plt.legend()
plt.ylim((1, 1.60))
#plt.savefig("azurlight_beam_after_telescope.png")

print(popt)