"""
Simulate screens observed by James and Hengrui in the case that the screens do
not interact with each other.

Simulation is of Screns C and D in James's draft.

s1 - SCREEN D
s2 - SCREEN C
"""
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import (
    CartesianRepresentation, CylindricalRepresentation,
    UnitSphericalRepresentation)

from screens.screen import Source, Screen1D, Telescope
from screens.fields import phasor


dp = 0.37*u.kpc
d2 = 0.0518*u.kpc
d1 = 0.0333*u.kpc

# This breaks the code, should maybe let Marten know
# dp = dp.to_value(u.pc) * u.pc
# d2 = d2.to_value(u.pc) * u.pc
# d1 = d1.to_value(u.pc) * u.pc


pulsar = Source(CartesianRepresentation([0., 0., 0.]*u.AU),
                vel=CartesianRepresentation(655.848, 0., 0., unit=u.km/u.s))
telescope = Telescope(CartesianRepresentation([0., 0., 0.]*u.AU))

s1 = Screen1D(CylindricalRepresentation(1., (10.3 - 11.4)*u.deg, 0.).to_cartesian(),
              1*np.array([-0.711, -0.62, -0.53, -0.304, -0.111, -0.052, -0.031, 0.0001, 0.0201, 0.0514, 0.102, 0.199, 0.3001, 0.409])*u.AU,
              magnification=np.array([0.01, 0.01, 0.02, 0.08, 0.25j, 0.34, 0.4+.1j,1, 0.2-.5j, 0.5j, 0.3, 0.2, 0.09, 0.02]))

s2 = Screen1D(CylindricalRepresentation(1., (-18.9 - 11.4)*u.deg, 0.).to_cartesian(),
              1.5*np.array([-0.711, -0.62, -0.53, -0.304, -0.111, -0.052, -0.031,0.0001, 0.0201, 0.0514, 0.102, 0.199, 0.3001, 0.409])*u.AU,
              magnification=np.array([0.01, 0.01, 0.02, 0.08, 0.25j, 0.34, 0.4+.1j, 1, 0.2-.5j, 0.5j, 0.3, 0.2, 0.09, 0.02]))


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
    # Connect origins
    # ax.plot(np.zeros(4), np.zeros(4),
    #         [0., d1.value, d2.value, dp.value], color='black')

    obs1 = telescope.observe(
        s1.observe(pulsar, distance=dp-d1),
        distance=d1)
    path_shape = obs1.tau.shape  # Also trigger calculation of pos, vel.
    tpos = obs1.pos
    scat1 = obs1.source.pos
    ppos = obs1.source.source.pos
    x = np.vstack(
        [np.broadcast_to(getattr(pos, 'x').to_value(u.AU), path_shape).ravel()
         for pos in (tpos, scat1, ppos)])
    y = np.vstack(
        [np.broadcast_to(getattr(pos, 'y').to_value(u.AU), path_shape).ravel()
         for pos in (tpos, scat1, ppos)])
    z = np.vstack(
        [np.broadcast_to(d, path_shape).ravel()
         for d in (0., d1.value, dp.value)])
    for _x, _y, _z in zip(x.T, y.T, z.T):
        ax.plot(_x, _y, _z, color='black', linestyle=':')
        ax.scatter(_x[1], _y[1], _z[1], marker='o',
                   color='red')
    obs2 = telescope.observe(
            s2.observe(pulsar, distance=dp-d2), distance=d2)
    path_shape = obs2.tau.shape  # Also trigger calculation of pos, vel.
    tpos = obs2.pos
    scat1 = obs2.source.pos
    ppos = obs2.source.source.pos
    x = np.vstack(
        [np.broadcast_to(getattr(pos, 'x').to_value(u.AU), path_shape).ravel()
         for pos in (tpos, scat1, ppos)])
    y = np.vstack(
        [np.broadcast_to(getattr(pos, 'y').to_value(u.AU), path_shape).ravel()
         for pos in (tpos, scat1, ppos)])
    z = np.vstack(
        [np.broadcast_to(d, path_shape).ravel()
         for d in (0., d2.value, dp.value)])
    for _x, _y, _z in zip(x.T, y.T, z.T):
        ax.plot(_x, _y, _z, color='black', linestyle=':')
        ax.scatter(_x[1], _y[1], _z[1], marker='o',
                   color=['orange'])

    # Create dynamic spectrum using delay for each path.
    tau0 = np.hstack([obs1.tau.ravel(), obs2.tau.ravel()])
    taudot = np.hstack([obs1.taudot.ravel(), obs2.taudot.ravel()])
    brightness = np.hstack([
        np.broadcast_to(obs1.brightness, obs1.tau.shape).ravel(),
        np.broadcast_to(obs2.brightness, obs2.tau.shape).ravel()])
    t = np.linspace(0, 120*u.min, 200)[:, np.newaxis]
    f = np.linspace(300*u.MHz, 310*u.MHz, 300)
    tau = (tau0[:, np.newaxis, np.newaxis]
           + taudot[:, np.newaxis, np.newaxis] * t)
    ph = phasor(f, tau)
    dw = ph * brightness[:, np.newaxis, np.newaxis]
    # Calculate and show dynamic spectrum.
    #ds = np.abs(dw.sum(0))**2
    ds = dw.sum(0)
    ax_ds = plt.subplot(233)
    ax_ds.imshow((np.abs(dw.sum(0))**2).T, cmap='Greys',
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
    ax_ss.set_xlim(-5, 5)
    ax_ss.set_ylim(-10, 10)
    ax_ss.set_xlabel(fd.unit.to_string('latex'))
    ax_ss.set_ylabel(tau.unit.to_string('latex'))

    plt.show()
