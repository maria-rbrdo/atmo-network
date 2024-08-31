import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
from scipy.signal import savgol_filter

tol = 1e-10
alpha = 0.3

deg = 1000  # no more than ten, weights problematic!
mu, w = roots_legendre(deg)
w = np.array(w)
mu = np.array(mu)

lat = np.rad2deg(np.arcsin(mu))
Omega = 2 * np.pi / (3600 * 24)  # rad/s
H = 10*1e3  # m
R = 6.371*1e6  # m
g = 9.81  # m/s

Phi = np.zeros_like(lat)
zeta = np.zeros_like(lat)
u = np.zeros_like(lat)

qs = 0.2933 * 2 * Omega / H
qv = 2 * Omega / H + qs
qe = lambda phi: (qv * (50 < phi <= 90) + qs * (0 < phi <= 50)
                  + 2 * Omega * np.sin(np.deg2rad(phi)) / H * (-90 <= phi <= 0) + 0)
qe = np.array([qe(l) for l in lat])
q0 = np.zeros_like(lat)

Phi[:] = H * g
it = 0
while True:
    # vorticity
    q0_num = 0
    q0_den = 0
    for i in range(0, deg):
        q0_num = q0_num + qe[i] * Phi[i] * w[i]
        q0_den = q0_den + Phi[i] * w[i]
    q0 = - q0_num / q0_den
    zeta_prev = zeta
    zeta = (qe + q0) * Phi / g - 2 * Omega * mu

    if it > 1:
        zeta = alpha * zeta + (1-alpha) * zeta_prev

    # velocity
    u[0] = zeta[0] * w[0]
    for i in range(1, deg):
        u[i] = u[i - 1] + zeta[i] * w[i]
    u = - R * u / np.sqrt(1-mu**2)
    uu = u / np.sqrt(1-mu**2)

    # geopotential height
    Phi[0] = uu[0] * mu[0] * (2*Omega*R+uu[0]) * w[0]
    for i in range(1, deg):
        Phi[i] = Phi[i - 1] + uu[i] * mu[i] * (2*Omega*R+uu[i]) * w[i]
    Phi = - Phi
    Phi_av = np.sum(Phi * w) / 2
    Phi = Phi - Phi_av + H * g

    # error
    q = (2*Omega*mu + zeta)*g/Phi - q0
    err = np.sum(np.sqrt((qe - q)**2))

    # iteration
    it += 1
    print(it, err)

    # break
    if err < tol:
        break

plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(15, 5))
ax = plt.subplot(1, 5, 1)
ax.plot(qe/Omega*H, lat, "k-")
ax.plot(2 * np.sin(np.deg2rad(lat)), lat, "k--")
ax.set_ylim([-90, 90])
ax.set_title("(a)")
ax.set_ylabel(r"$\varphi$ (deg)")
ax.set_xlabel(r"$q_eH/\Omega$ (unitless)", labelpad=15)
ax = plt.subplot(1, 5, 2)
ax.plot(zeta, lat, "k-")
ax.plot(np.zeros_like(lat), lat, "k--")
ax.set_ylim([-90, 90])
ax.set_title("(b)")
ax.set_yticklabels([])
ax.set_xlabel(r"$\zeta$ (s$^{-1}$)", labelpad=15)
ax = plt.subplot(1, 5, 3)
ax.plot(u, lat, "k-")
ax.plot(np.zeros_like(lat), lat, "k--")
ax.set_ylim([-90, 90])
ax.set_xlim([-99, 99])
ax.set_title("(c)")
ax.set_yticklabels([])
ax.set_xlabel("$u$ (m s$^{-1}$)", labelpad=15)
ax = plt.subplot(1, 5, 4)
ax.plot(Phi/(H*g)-1, lat, "k-")
ax.plot(np.zeros_like(lat), lat, "k--")
ax.set_ylim([-90, 90])
ax.set_xlim([-0.19, 0.19])
ax.set_title("(d)")
ax.set_yticklabels([])
ax.set_xlabel(r"$\Phi_e'/\langle \Phi_e \rangle$ (unitless)", labelpad=15)
fig.subplots_adjust(wspace=0.35)

# DRAG

r0 = 1/20
r = lambda phi: r0 * (1 - np.tanh((phi-np.pi/18)/(np.pi/18)))
ax = plt.subplot(1, 5, 5)
ax.plot([r(np.deg2rad(l)) for l in lat], lat, "k-")
ax.plot(np.zeros_like(lat), lat, "k--")
ax.set_ylim([-90, 90])
ax.set_xlim([-0.02, 0.12])
ax.set_title("(e)")
ax.set_yticklabels([])
ax.set_xlabel(r"$r$ (d$^{-1}$)", labelpad=15)
fig.subplots_adjust(left=0.07, right=0.95, top=0.9, bottom=0.2, wspace=0)

fig.savefig("settings.png", dpi=300)