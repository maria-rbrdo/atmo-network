import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Parameters ...........................................................................................................
A = 0.1
eps = 0
omega = 2 * np.pi / 10

nspace = 500
ntime = 15
dt = 0.1

# Files ................................................................................................................
folder = os.path.expanduser("../../dataloc/quadgyre/netdata/")
upath = os.path.expanduser(f"../../dataloc/quadgyre/netdata/u_e{eps}_s{nspace}_t{ntime}")
vpath = os.path.expanduser(f"../../dataloc/quadgyre/netdata/v_e{eps}_s{nspace}_t{ntime}")

if not os.path.exists(folder):
    os.mkdir(folder)
try:
    os.remove(upath)
    print(f"Previous file '{upath}' deleted successfully.")
except:
    pass

try:
    os.remove(vpath)
    print(f"Previous file '{vpath}' deleted successfully.")
except:
    pass

# Get data .............................................................................................................
a = lambda t: eps * np.sin(omega * t)
b = lambda t: 1 - 2 * eps * np.sin(omega * t)
f = lambda x, t: a(t) * (x**2) + b(t) * x
df = lambda x, t: 2 * a(t) * x + b(t)
u = lambda x, y, t: - np.pi * A * np.sin(np.pi * f(x, t)) * np.cos(np.pi * y)
v = lambda x, y, t: np.pi * A * np.cos(np.pi * f(x, t)) * np.sin(np.pi * y) * df(x, y)

xx = np.linspace(0, 2, nspace)
yy = np.linspace(-1, 1, nspace)
tt = np.arange(0, ntime, dt)

X, Y, T = np.meshgrid(xx, yy, tt)

U = u(X, Y, T)
V = v(X, Y, T)

plt.quiver(X[:, :, 0], Y[:, :, 0], U[:, :, 0], V[:, :, 0])
plt.show()

# Store data ...........................................................................................................

with h5py.File(upath, mode='a') as fu, h5py.File(vpath, mode='a') as fv:
    fu.create_dataset("data", data=U)
    fu.create_dataset("x", data=xx)
    fu.create_dataset("y", data=yy)
    fu.create_dataset("time", data=tt)

    fv.create_dataset("data", data=V)
    fv.create_dataset("x", data=xx)
    fv.create_dataset("y", data=yy)
    fv.create_dataset("time", data=tt)
