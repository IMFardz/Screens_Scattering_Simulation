# Licensed under the GPLv3 - see LICENSE
"""Simulate scintillation through two screens.
The setup is somewhat similar to what is seen in the Brisken data,
with one screen that has a lot of images, and another that has only
one.
The geometry of the paths is shown, as well as inferred dynamic and
secondary spectra.
.. warning:: Usage quite likely to change.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import (
    CartesianRepresentation, CylindricalRepresentation,
    UnitSphericalRepresentation)

from screens.screen import Source, Screen1D, Telescope
from screens.fields import phasor
import astropy.constants as ac

# Distances
dp = 0.75*u.kpc
d2 = 0.50*u.kpc
d1 = 0.25*u.kpc

# Angles
ap =  0* u.deg      # Pulsar Angle
a1 = 90* u.deg      # Screen 1 Angle
a2 = 45* u.deg      # Screen 2 Angle

pulsar = Source(CartesianRepresentation([0., 0., 0.]*u.AU),
                vel=CartesianRepresentation(300., 0., 0., unit=u.km/u.s))
telescope = Telescope(CartesianRepresentation([0., 0., 0.]*u.AU),
        vel=CylindricalRepresentation(0, 0*u.deg, 0.).to_cartesian()*u.km/u.s)

s1 = Screen1D(CylindricalRepresentation(1., a1, 0.).to_cartesian(),
              [-0.711, -0.62, -0.53, -0.304, -0.111, -0.052, -0.031,
               0.0001, 0.0201, 0.0514, 0.102, 0.199, 0.3001, 0.409]*u.AU,
               v=0*u.km/u.s,
               magnification=np.array(
                  [0.01, 0.01, 0.02, 0.08, 0.25j, 0.34, 0.4+.1j,
                   1, 0.2-.5j, 0.5j, 0.3, 0.2, 0.09, 0.02]))
s2 = Screen1D(CylindricalRepresentation(1., a2, 0.).to_cartesian(),
               p=[1e-9]*u.m, v=0*u.km/u.s, magnification=0.05)


# Compute the curvature one would observe is just single screen at each location
s_1p = 1 - (d1/dp).to(u.dimensionless_unscaled)
s_2p = 1 - (d2/dp).to(u.dimensionless_unscaled)
s_3 = 1 - ((d2-d1)/(dp-d1)).to(u.dimensionless_unscaled)
s_12 = 1 - (d1/d2).to(u.dimensionless_unscaled)
deff1 = dp * (1-s_1p)/s_1p
deff2 = dp * (1-s_2p)/s_2p
deff3 = (dp-d1) * (1-s_3)/s_3
deff4 = dp * (1 - s_1p)/s_12
veff1 = np.dot(pulsar.vel.get_xyz() * (1 - s_1p)/s_1p + telescope.vel.get_xyz() - s1.v/s_1p, s1.normal.get_xyz())
veff2 = np.dot(pulsar.vel.get_xyz() * (1 - s_2p)/s_2p + telescope.vel.get_xyz() - s2.v/s_2p, s2.normal.get_xyz())
veff4 = veff1 * np.sqrt(1 - np.dot(s1.normal, s2.normal)**2)**2
lambd = ac.c / (305 * u.MHz)
eta1 = ((lambd**2/(2*ac.c)) * deff1/(veff1**2)).to(u.s**3)
eta2 = ((lambd**2/(2*ac.c)) * deff2/(veff2**2)).to(u.s**3)
eta4 = ((lambd**2/(2*ac.c)) * deff1/(veff4**2)).to(u.s**3)

psr_vel  = np.sqrt(np.dot(pulsar.vel.get_xyz(), pulsar.vel.get_xyz()))
psr_vel2 = np.sin(ap - a2) * psr_vel
psr_vel3 = np.sin(a2 - a1) * psr_vel2 * s1.normal.get_xyz()

s2_v = np.sqrt(1 - np.dot(s2.normal.get_xyz(), s1.normal.get_xyz())**2) * s2.v * s1.normal.get_xyz()

veff= np.dot(psr_vel3 * (1 - s_1p)/s_1p + telescope.vel.get_xyz() + s1.v/s_1p * s1.normal.get_xyz() - s2_v, s1.normal.get_xyz())
deff=deff1
eta = ((lambd**2/(2*ac.c)) * deff/(veff**2)).to(u.s**3)
print("veff = {:.5f}".format(veff))
print("eta = {:.5f}".format(eta))

# Compute offset in second parabola
offset_x = -(veff2/(lambd)*(s2.p/d2)).to(u.mHz)
offset_y = ((deff2/(2*ac.c))*(s2.p/d2)**2).to(u.us)
#print(offset_x)
#print(offset_y)

# Compute position of point
theta = (s2.p[0]/d2).to(u.dimensionless_unscaled)
fd_theta = (veff2 * theta / lambd).to(u.mHz)
tau_theta = (deff2 * theta**2 / (2*ac.c)).to(u.us)
print(fd_theta)
print(tau_theta)


def axis_extent(x):
    x = x.ravel().value
    dx = x[1]-x[0]
    return x[0]-0.5*dx, x[-1]+0.5*dx


def unit_vector(c):
    return c.represent_as(UnitSphericalRepresentation).to_cartesian()


ZHAT = CartesianRepresentation(0., 0., 1., unit=u.one)


def plot_screen(ax, s, d, color='black', **kwargs):
    d = d.to_value(u.kpc)
    x = np.array(ax.get_xlim3d())
    y = np.array(ax.get_ylim3d())[:, np.newaxis]
    ax.plot_surface([[-2.1, 2.1]]*2, [[-2.1]*2, [2.1]*2], d*np.ones((2, 2)),
                    alpha=0.1, color=color)
    x = ax.get_xticks()
    y = ax.get_yticks()[:, np.newaxis]
    ax.plot_wireframe(x, y, np.broadcast_to(d, (x+y).shape),
                      alpha=0.2, color=color)
    spos = s.normal * s.p if isinstance(s, Screen1D) else s.pos
    ax.scatter(spos.x.to_value(u.AU), spos.y.to_value(u.AU),
               d, c=color, marker='+')
    if spos.shape:
        for pos in spos:
            zo = np.arange(2)
            ax.plot(pos.x.to_value(u.AU)*zo, pos.y.to_value(u.AU)*zo,
                    np.ones(2) * d, c=color, linestyle=':')
            upos = pos + (ZHAT.cross(unit_vector(pos))
                          * ([-1.5, 1.5] * u.AU))
            ax.plot(upos.x.to_value(u.AU), upos.y.to_value(u.AU),
                    np.ones(2) * d, c=color, linestyle='-')
    elif s.vel.norm() != 0:
        dp = s.vel * 5 * u.day
        ax.quiver(spos.x.to_value(u.AU), spos.y.to_value(u.AU), d,
                  dp.x.to_value(u.AU), dp.y.to_value(u.AU), np.zeros(1),
                  arrow_length_ratio=0.05)


if __name__ == '__main__':
    print_check = False

    fig = plt.figure()
    fig.tight_layout()
    ax = plt.subplot(1, 3, (1, 2), projection='3d')
    ax.set_box_aspect((1, 1, 2))
    ax.set_axis_off()
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_xticks([-2, -1, 0, 1., 2])
    ax.set_yticks([-2, -1, 0, 1., 2])
    ax.set_zticks([0, 0.25, 0.5, 0.75])
    plot_screen(ax, telescope, 0*u.kpc, color='blue')
    plot_screen(ax, s1, d1, color='red')
    plot_screen(ax, s2, d2, color='orange')
    plot_screen(ax, pulsar, dp, color='green')


    obs2 = telescope.observe(
        s1.observe(
            s2.observe(pulsar, distance=dp-d2),
            distance=d2-d1),
        distance=d1)
    path_shape = obs2.tau.shape  # Also trigger calculation of pos, vel.
    tpos = obs2.pos
    scat1 = obs2.source.pos
    scat2 = obs2.source.source.pos
    ppos = obs2.source.source.source.pos
    x = np.vstack(
        [np.broadcast_to(getattr(pos, 'x').to_value(u.AU), path_shape).ravel()
         for pos in (tpos, scat1, scat2, ppos)])
    y = np.vstack(
        [np.broadcast_to(getattr(pos, 'y').to_value(u.AU), path_shape).ravel()
         for pos in (tpos, scat1, scat2, ppos)])
    z = np.vstack(
        [np.broadcast_to(d, path_shape).ravel()
         for d in (0., d1.value, d2.value, dp.value)])
    for _x, _y, _z in zip(x.T, y.T, z.T):
        ax.plot(_x, _y, _z, color='black', linestyle=':')
        ax.scatter(_x[1:3], _y[1:3], _z[1:3], marker='o',
                   color=['red', 'orange'])

    # Create dynamic spectrum using delay for each path.
    tau0 = np.hstack([obs2.tau.ravel()])
    taudot = np.hstack([obs2.taudot.ravel()])
    brightness = np.hstack([
        np.broadcast_to(obs2.brightness, obs2.tau.shape).ravel()])
    t = np.linspace(0, 120*u.min, 200)[:, np.newaxis]
    f = np.linspace(300*u.MHz, 310*u.MHz, 300)
    tau = (tau0[:, np.newaxis, np.newaxis]
           + taudot[:, np.newaxis, np.newaxis] * t)
    ph = phasor(f, tau)
    dw = ph * brightness[:, np.newaxis, np.newaxis]
    # Calculate and show dynamic spectrum.
    ds = dw.sum(0)
    #ds = np.abs(dw.sum(0))**2
    ax_ds = plt.subplot(233)
    ax_ds.imshow(np.abs(ds.T), cmap='Greys',
                 extent=axis_extent(t) + axis_extent(f),
                 origin='lower', interpolation='none', aspect='auto')
    ax_ds.set_xlabel(t.unit.to_string('latex'))
    ax_ds.set_ylabel(f.unit.to_string('latex'))
    # And the conjugate spectrum.
    ss = np.fft.fft2(ds)
    ss /= ss[0, 0]
    ss = np.fft.fftshift(ss)
    tau = np.fft.fftshift(np.fft.fftfreq(f.size, f[1]-f[0])).to(u.us)
    fd = np.fft.fftshift(np.fft.fftfreq(t.size, t[1]-t[0])).to(u.mHz)
    ax_ss = plt.subplot(236)
    ax_ss.imshow(np.log10(np.abs(ss.T)**2), vmin=-7, vmax=0, cmap='Greys',
                 extent=axis_extent(fd) + axis_extent(tau),
                 origin='lower', interpolation='none', aspect='auto')
    ax_ss.plot(fd, eta*fd**2, color='red')
    ax_ss.set_xlim(-5, 5)
    ax_ss.set_ylim(-10, 10)
    ax_ss.set_xlabel(fd.unit.to_string('latex'))
    ax_ss.set_ylabel(tau.unit.to_string('latex'))

    plt.show()
    #plt.close()

    fig = plt.figure(figsize=(20, 6))
    fig.add_subplot(121)
    plt.imshow(np.abs(ds.T), cmap='Greys',
                 extent=axis_extent(t) + axis_extent(f),
                 origin='lower', interpolation='none', aspect='auto')
    plt.xlabel(t.unit.to_string('latex'))
    plt.ylabel(f.unit.to_string('latex'))
    plt.colorbar()

    fig.add_subplot(122)
    plt.imshow(np.log10(np.abs(ss.T)**2), vmin=-7, vmax=0, cmap='Greys',
                 extent=axis_extent(fd) + axis_extent(tau),
                 origin='lower', interpolation='none', aspect='auto')
    plt.plot(fd, eta*fd**2, color='red')
    #plt.plot(fd, eta1*fd**2, color='blue')
    plt.xlim(-5, 5)
    plt.ylim(0, 10)
    plt.xlabel(fd.unit.to_string('latex'))
    plt.ylabel(tau.unit.to_string('latex'))
    plt.colorbar()
    plt.tight_layout()
    plt.show()
