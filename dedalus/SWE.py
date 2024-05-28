"""
ENVIRONMENT: dedalus3

Dedalus script simulating the shallow water equations on a sphere. Data
is saved to HDF5 files in the folder data/model which must exist.

The script implements the test case of a barotropically unstable mid-latitude
jet from Galewsky et al. 2004 (https://doi.org/10.3402/tellusa.v56i5.14436).
The initial height field balanced the imposed jet is solved with an LBVP.
A perturbation is then added and the solution is evolved as an IVP.

To run using e.g. 4 processes:
    $ mpiexec -n 4 python3 SWE.py
"""

import numpy as np
from scipy import integrate
import dedalus.public as d3
from alive_progress import alive_bar
import logging
logger = logging.getLogger(__name__)

#%% Definitions

# Simulation units:
meter = 1 / 6.37122e6  # -> 1 dist unit is 1 Earth radius
hour = 1  # -> 1 time unit is 1 hour.
second = hour / 3600

# Model parameters:
R = 6.37122e6 * meter  # radius of the Earth = 1
Omega = 7.292e-5 / second  # angular velocity of the Earth
nu = 1e5 * meter**2 / second # diffusion coefficient
nu_hp = nu / 32**2   # hyperdiffusion matched at ell=32
g = 9.80616 * meter / second**2  # gravitational acceleration
H = 1e4 * meter  # height atmosphere

# Simulation parameters:
timestep = 60 * second  # 1 min hence 60 time steps per hour
stop_sim_time = 360 * hour

# Spectral parameters:
nmodes = 2**6  # number of spectral modes (powers of two are computationally efficient, especially for FFT algorithms)
ntheta = 2 * nmodes  # covers 360 degrees
nphi = nmodes  # covers 180 degrees
dealias = 3/2  # needed to dealias quadratic nonlinearities

# Data type:
dtype = np.float64

# Bases
coords = d3.S2Coordinates('theta', 'phi')
dist = d3.Distributor(coords, dtype=dtype)
basis = d3.SphereBasis(coords, (ntheta, nphi), radius=R, dealias=dealias, dtype=dtype)
theta, phi = dist.local_grids(basis)
lat = np.pi / 2 - phi + 0 * theta  # define latitude
lon = np.pi - 0 * phi - theta  # define longitude

# Fields
u = dist.VectorField(coords, name='u', bases=basis)  # velocity field
h = dist.Field(name='h', bases=basis)  # height field

# Substitutions
zcross = lambda A: d3.MulCosine(d3.skew(A))  # cosine multip. for S2 (90-degree positive rotation of 2D vector field)

#%% Initial conditions: zonal jet

umax = 80 * meter / second  # maximum zonal velocity
lat0 = np.pi / 7  # latitude southern boundary
lat1 = np.pi / 2 - lat0  # latitude northern boundary
en = np.exp(-4 / (lat1 - lat0)**2)  # normalising parameter

jet = (lat0 <= lat) * (lat <= lat1)  # jet domain
u_jet = umax / en * np.exp(1 / (lat[jet] - lat0) / (lat[jet] - lat1))  # jet
u['g'][0, jet] = u_jet  # set initial condition

#%% Initial conditions: balanced height
def integrand(x):  # define integrand of the integral
    if lat0 > x or lat1 < x:
        return 0
    else:
        u_jett = umax / en * np.exp(1 / (x - lat0) / (x - lat1))
        return R * u_jett * (2 * Omega * np.sin(x) + np.tan(x) * u_jett / R)
cumulative_integrate = np.vectorize(lambda x: integrate.quad(integrand, 0.0, x, epsabs=1e-15, epsrel=1e-15)[0])
h_int = cumulative_integrate(lat[0, :].squeeze())  # evaluate integral
h_0 = - h_int/g + np.mean(h_int)/g  # find height profile
h['g'] = np.vstack([h_0] * ntheta)  # set height profile around the globe

#%% Initial conditions: perturbation
lat2 = np.pi / 4  # latitude perturbation
hpert = 120 * meter  # perturbation amplitude
alpha = 1 / 3  # perturbation parameter
beta = 1 / 15  # perturbation parameter
h['g'] += hpert * np.cos(lat) * np.exp(-(lon/alpha)**2) * np.exp(-((lat2-lat)/beta)**2)  # set perturbation

#%% Problem
problem = d3.IVP([u, h], namespace=locals())
# problem.add_equation("dt(u) - nu*lap(u) + nu_hp*lap(lap(u)) + g*grad(h) + 2*Omega*zcross(u) = - u@grad(u)")
# problem.add_equation("dt(h) - nu*lap(h) + nu_hp*lap(lap(h)) + H*div(u) = - div(h*u)")
problem.add_equation("dt(u) + nu_hp*lap(lap(u)) + g*grad(h) + 2*Omega*zcross(u) = - u@grad(u)")
problem.add_equation("dt(h) + nu_hp*lap(lap(h)) + H*div(u) = - div(h*u)")

#%% Solver
solver = problem.build_solver(d3.CNAB2)
solver.stop_sim_time = stop_sim_time

#%% Analysis
snapshots = solver.evaluator.add_file_handler('data/model/SWE_snapshots', iter=int(10 * 60 * second / timestep),
                                              sim_dt=stop_sim_time, max_writes=5000)  # save data every 10 min
snapshots.add_task(h, name='height')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
snapshots.add_task(u, name='velocity')

#%% Main loop
try:
    logger.info('Starting main loop')
    with alive_bar(manual=True, force_tty=True) as bar:
        while solver.proceed:
            solver.step(timestep)
            bar(solver.sim_time/solver.stop_sim_time)
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

