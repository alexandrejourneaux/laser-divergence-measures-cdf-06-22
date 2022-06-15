import numpy as np
from scipy.optimize import minimize

first_test = False

#%% Experimental parameters, all length are given in mm

# From measurements on the laser:
waist_in = 0.5353925373568325 # mm
z0_in = 171.13369181371924 # mm
wavelength = 1050.37e-6 # mm

zR_in = np.pi * waist_in**2 / wavelength

# Available lenses (the third lens in imposed),
# -100 and 175 mm focal lenths are already used for the 1550 nm setup:
available_focals = [400, 100, 40, -75, 100, 200, -100, 175]
f3 = 200

# A hyper-parameter between 0 and 1 defines the importance of the Rayleigh 
# length vs the importance of the distance to the last lens (waist in focal plane)
# alpha = 1 : we don’t care about the lens-waist distance
# alpha = 0 : we don’t care about the Rayleigh length
alpha = 0.999

#%% Useful functions

def q_param(z, z0, zR):
    return (z - z0) + 1j*zR

def apply(mat, q_in):
    return (mat[0,0]*q_in + mat[0,1]) / (mat[1,0]*q_in + mat[1,1])

def lens(f):
    return np.matrix([[1, 0],
                      [-1/f, 1]])

def propagation(d):
    return np.matrix([[1, d],
                      [0, 1]])

def q_output(z_in, z0_in, zR_in, d01, f1, d12, f2, d23):
    q = q_param(z_in, z0_in, zR_in)
    q = apply(propagation(d01), q)
    q = apply(lens(f1), q)
    q = apply(propagation(d12), q)
    q = apply(lens(f2), q)
    q = apply(propagation(d23), q)
    q = apply(lens(f3), q)
    
    return q

def telescope_output(z_in, z0_in, zR_in, d01, f1, d12, f2, d23):
    q = q_param(z_in, z0_in, zR_in)
    q = apply(propagation(d01), q)
    q = apply(lens(f1), q)
    print(q)
    q = apply(propagation(d12), q)
    q = apply(lens(f2), q)
    
    return q

def zR_output(z_in, z0_in, zR_in, d01, f1, d12, f2, d23):
    return q_output(z_in, z0_in, zR_in, d01, f1, d12, f2, d23).imag

def to_minimize(x, zR_opt, f1, f2):
    """ x = array([d01, d12])"""
    
    d01 = x[0]
    d12 = x[1]
    d23 = x[2]
    
    q_out = q_output(z_in, z0_in, zR_in, d01, f1, d12, f2, d23)
    
    return np.abs(alpha * (q_out.imag - zR_opt)**2 + (1 - alpha) * (q_out.real + f3)**2)

#%% First test with manually fixed f1 and f2

if first_test:
    z_in = 0
    
    f1 = 100
    f2 = 175
    
    d01_guess = 100
    d12_guess = f1 + f2
    d23_guess = 100
    
    print(minimize(to_minimize,
                   np.array([d01_guess, d12_guess, d23_guess]), 
                   args=(7, f1, f2),
                   bounds=((0, np.inf), (0, np.inf), (0, np.inf))
                   )
          )
    
    d01, d12, d23 = minimize(to_minimize,
                             np.array([d01_guess, d12_guess, d23_guess]),
                             args=(7, f1, f2),
                             bounds=((0, np.inf), (0, np.inf), (0, np.inf))
                             )["x"]
    print(q_output(z_in, z0_in, zR_in, d01, f1, d12, f2, d23))

#%% Optimization routine

z_in = 0
zR_opt = 7
min_d01 = 40
max_d01 = 200
min_d12 = 20
max_d12 = 150
min_d23 = 200
max_d23 = 300
d01_guess = 100
d12_guess = 100
d23_guess = 100

    
N = len(available_focals)
total_dist_array = np.empty((N, N))
d01_array = np.empty((N, N))
d12_array = np.empty((N, N))
d23_array = np.empty((N, N))
dist_to_waist_array = np.empty((N, N))
rayleigh_array = np.empty((N, N))
validity_array = np.empty((N, N))

for i, f1 in enumerate(available_focals):
    for j, f2 in enumerate(available_focals):
        #We do not allow to use twice the same lens:
        if i == j:
            total_dist_array[i, i] = np.inf
            d01_array[i, i] = np.inf
            d12_array[i, i] = np.inf
            d23_array[i, i] = np.inf
            dist_to_waist_array[i, i] = np.inf
            rayleigh_array[i, i] = np.inf
            validity_array[i, i] = np.inf
            
            
        else:
            sol = minimize(to_minimize,
                           np.array([d01_guess, d12_guess, d23_guess]),
                           args=(zR_opt, f1, f2),
                           bounds=((min_d01, max_d01), (min_d12, max_d12), (min_d23, max_d23))
                           )
            
            d01, d12, d23 = sol["x"]
            validity = sol["fun"]
            
            dist_to_waist = - q_output(z_in, z0_in, zR_in, d01, f1, d12, f2, d23).real
            waist = q_output(z_in, z0_in, zR_in, d01, f1, d12, f2, d23).imag
            
            if dist_to_waist < 0:
                total_dist_array[i, j] = np.inf
                d01_array[i, j] = np.inf
                d12_array[i, j] = np.inf
                d23_array[i, j] = np.inf
                dist_to_waist_array[i, j] = np.inf
                rayleigh_array[i, j] = np.inf
                validity_array[i, j] = np.inf

            
            else:
                total_dist_array[i, j] = d01 + d12 + d23 + dist_to_waist
                d01_array[i, j] = d01
                d12_array[i, j] = d12
                d23_array[i, j] = d23
                dist_to_waist_array[i, j] = dist_to_waist
                rayleigh_array[i, j] = waist
                validity_array[i, j] = validity

min_index = np.unravel_index(validity_array.argmin(), total_dist_array.shape)
best_setups = [np.unravel_index(x, validity_array.shape)for x in np.argsort(validity_array.reshape((-1,)))]

# f1_opt = available_focals[best_setups[0][0]]
# f2_opt = available_focals[best_setups[0][1]]

for i in range(3):
    print("\nBest setup n°", i+1, " @1050nm \n-----------------------")
    print("f1 = ", available_focals[best_setups[i][0]], " mm")
    print("f2 = ", available_focals[best_setups[i][1]], " mm")
    print("d01 = ", d01_array[best_setups[i]], " mm")
    print("d12 = ", d12_array[best_setups[i]], " mm")
    print("d23 = ", d23_array[best_setups[i]], " mm")
    print("Distance from last lens to waist = ", dist_to_waist_array[best_setups[i]], " mm")
    print("Total setup length = ", total_dist_array[best_setups[i]], " mm")
    print("Resulting Rayleigh length = ", rayleigh_array[best_setups[i]], " mm")
              
#%%  Best result fine tuning

f1 = 40
f2 = 100
d01 =  195
d12 =  139
d23 =  470

print(q_output(z_in, z0_in, zR_in, d01, f1, d12, f2, d23))
    
     
