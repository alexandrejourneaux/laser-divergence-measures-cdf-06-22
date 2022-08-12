import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def dx(N):
    res = np.diag(np.ones((N-1,), dtype=float), k=1) - np.diag(np.ones((N-1,), dtype=float), k=-1)
    res[0, :3] = [-3, 4, -1]
    res[-1, -3:] = [1, -4, 3]
    
    return res

def gaussian(x, amp, avg, w):
    return amp * np.exp(- 2*(x - avg)**2 / w**2)


z = np.arange(6.10, 6.80, 0.02) # mm
power = np.array([4.95, 4.95, 4.96, 4.96, 4.96, 4.95, 4.94, 4.93, 4.92, 4.91,
                  4.90, 4.83, 4.63, 4.17, 3.27, 2.30, 1.13, 0.462, 0.112,
                  0.0260, 0.0126, 0.0112, 0.0102, 0.00944, 0.00914, 0.00882,
                  0.00830, 0.00805, 0.00781, 0.00754, 0.00732, 0.00712,
                  0.00679, 0.00646, 0.00609, 0.00580])

beam_shape = - dx(len(power))@power

popt, pcov = curve_fit(gaussian, z, beam_shape, p0=[1, 5, 1], bounds=([-np.inf, -np.inf, 0],[np.inf, np.inf, np.inf]))

plt.figure()
plt.scatter(z, beam_shape)
z_finer = np.linspace(z[0], z[-1], 500)
plt.plot(z_finer, gaussian(z_finer, *popt), c="r")
print(popt)
