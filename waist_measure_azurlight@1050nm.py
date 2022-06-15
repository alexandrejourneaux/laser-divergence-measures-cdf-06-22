import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

wavelength = 1050.34e-9 # m

def waist(z, w0, z_offset):
    zR = np.pi * w0**2 / wavelength
    return w0 * np.sqrt(1 + ((z - z_offset)/zR)**2)

z_array_old = np.array([27, 45, 62, 82, 102, 122, 142])*1e-2
w_array_old = np.array([871, 928, 994, 1086, 1187, 1270, 1355])*1e-6  / 2 

z_array = np.array([54, 72, 92, 112, 132]) * 1e-2 
w_array_low = np.array([1078, 1158, 1340, 1535 ,1731]) * 1e-6 / 2
w_array_high = np.array([1250, 1289, 1380, 1575, 1830]) * 1e-6 / 2

w_array = np.sqrt(w_array_low * w_array_high)


doc_z_array = np.array([60, 90, 120, 150, 180, 210])*1e-2
doc_w_array = np.array([1245, 1433, 1723, 1978, 2343, 2678])*1e-6 / 2
                   
popt_low, pcov_low = curve_fit(waist, z_array, w_array_low, p0=[1e-3, 0], bounds=((0, -np.inf), (np.inf, np.inf)))
popt_high, pcov_high = curve_fit(waist, z_array, w_array_high, p0=[1e-3, 0], bounds=((0, -np.inf), (np.inf, np.inf)))

popt, pcov = curve_fit(waist, z_array, w_array, p0=[1e-3, 0], bounds=((0, -np.inf), (np.inf, np.inf)))

doc_popt, doc_pcov = curve_fit(waist, doc_z_array, doc_w_array, p0=[1e-3, 0], bounds=((0, -np.inf), (np.inf, np.inf)))

plt.figure()
# plt.scatter(z_array, w_array_low, c="b", label="My data, w0 = "+f"{popt_low[0]:.2e}")
# plt.scatter(z_array, w_array_high, c="b", label="My data, w0 = "+f"{popt_high[0]:.2e}")
plt.scatter(z_array, w_array, c="b", label="My data, w0 = "+f"{popt[0]:.2e}")
plt.scatter(doc_z_array, doc_w_array, c="g", label=f"Doc data, w0 = "+f"{doc_popt[0]:.2e}")

z_plot = np.linspace(-1, 3, 300)
# plt.plot(z_plot, [waist(z, *popt_low) for z in z_plot], c="r")
# plt.plot(z_plot, [waist(z, *popt_high) for z in z_plot], c="r")
plt.plot(z_plot, [waist(z, *popt) for z in z_plot], c="r")
plt.plot(z_plot, [waist(z, *doc_popt) for z in z_plot], c="r")

plt.legend()

print("waist = ", popt[0]*1e3, " mm")