"""
Description of model: Two Parallel Screens, 1 images on first,
two images (not symmetrc) on second, only Earth observatories are moving
"""
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import (
    CartesianRepresentation, CylindricalRepresentation,
    UnitSphericalRepresentation)

from screens.screen import Source, Screen1D, Telescope
from screens.fields import phasor
from scipy.optimize import curve_fit
import astropy.constants as ac

# Distances
dp = 0.2*u.kpc
d2 = 0.1*u.kpc

# Angles
ap = 0 * u.deg
a2 = 130 * u.deg

# ON JANUARY 27, 2019 the observatory velocities are given by:
# ARECIBO: <[ 18.44724718, -13.00367214] km / s (22.56981209 km / s, -125.18043988 deg)
# JODRELL: <[ 18.75964149, -13.07485774] km / s (22.8664832  km / s, -124.87523808 deg)
# VLA    : <[ 18.60568632, -12.93659457] km / s (22.66113507 km / s, -124.81101557 deg)

pulsar = Source(CartesianRepresentation([0., 0., 0.]*u.AU),
                vel=CartesianRepresentation(300., 0., 0., unit=u.km/u.s))

arecibo = Telescope(CartesianRepresentation([0., 0., 0.]*u.AU),
        vel=CylindricalRepresentation(0, 0*u.deg, 0.).to_cartesian()*u.km/u.s)

jodrell = Telescope(CylindricalRepresentation(5000, a2, 0.).to_cartesian() * u.km,
        vel=CylindricalRepresentation(0, 0*u.deg, 0.).to_cartesian()*u.km/u.s)

vla = Telescope(CylindricalRepresentation(5000, a2 + 90*u.deg, 0.).to_cartesian() * u.km)

s2 = Screen1D(CylindricalRepresentation(1., a2, 0.).to_cartesian(),
              0.5*np.array([-0.34, -0.3, -0.25, -0.15, -0.06, -0.03, -0.02, 0.0001, 0.0201, 0.03, 0.05, 0.1, 0.15, 0.2])*u.AU,
              v=0*u.km/u.s,
              magnification=np.array([0.01, 0.01, 0.02, 0.08, 0.25j, 0.34, 0.4+.1j,1, 0.2-.5j, 0.5j, 0.3, 0.2, 0.09, 0.02]))


# Compute the curvature one would observe is just single screen at each location
s_2p = 1 - (d2/dp).to(u.dimensionless_unscaled)
deff = dp * (1-s_2p)/s_2p
veff = np.dot(pulsar.vel.get_xyz() * (1 - s_2p)/s_2p + arecibo.vel.get_xyz(), s2.normal.get_xyz())
lambd = ac.c / (305 * u.MHz)
eta = ((lambd**2/(2*ac.c)) * deff/(veff**2)).to(u.s**3)
print("veff = {:.5f}".format(veff))
print("eta = {:.5f}".format(eta))

# Predict the delay
jodrell_b = np.linalg.norm( jodrell.pos.get_xyz())
jodrell_u = jodrell.pos.get_xyz() / jodrell_b
vla_b = np.linalg.norm( vla.pos.get_xyz())
vla_u = vla.pos.get_xyz() / jodrell_b

delay_jb = -( jodrell_b / (veff / np.dot(jodrell_u, s2.normal.get_xyz()))).to(u.s)
delay_vla = -( vla_b / (veff / np.dot(vla_u, s2.normal.get_xyz()))).to(u.s)
print("Delay = {:.5f}".format(delay_jb))
print("Delay = {:.5f}".format(delay_vla))


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
    plot_screen(ax, arecibo, 0*u.kpc, color='blue')
    plot_screen(ax, s2, d2, color='orange')
    plot_screen(ax, pulsar, dp, color='green')


    # ARECIBO
    obs_ar = arecibo.observe(s2.observe(pulsar, distance=dp-d2), distance=d2)
    path_shape = obs_ar.tau.shape  # Also trigger calculation of pos, vel.
    tpos = obs_ar.pos
    scat1 = obs_ar.source.pos
    ppos = obs_ar.source.source.pos
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
        ax.scatter(_x[1:3], _y[1:3], _z[1:3], marker='o',
                   color=['red', 'orange'])

    # Create dynamic spectrum using delay for each path.
    tau0 = np.hstack([obs_ar.tau.ravel()])
    taudot = np.hstack([obs_ar.taudot.ravel()])
    brightness = np.hstack([np.broadcast_to(obs_ar.brightness, obs_ar.tau.shape).ravel()])
    t = np.linspace(0, 120*u.min, 300)[:, np.newaxis]
    f = np.linspace(300*u.MHz, 310*u.MHz, 600)
    tau = (tau0[:, np.newaxis, np.newaxis]
           + taudot[:, np.newaxis, np.newaxis] * t)
    ph = phasor(f, tau)
    dw = ph * brightness[:, np.newaxis, np.newaxis]
    # Calculate and show dynamic spectrum.
    #ds = np.abs(dw.sum(0))**2
    #ar_ds = (np.abs(dw.sum(0))**2).T
    ar_ds = (dw.sum(0)).T
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
    ax_ss.imshow(np.log10(np.abs(ss.T)**2), vmin=-7, vmax=5, cmap='Greys',
                 extent=axis_extent(fd) + axis_extent(tau),
                 origin='lower', interpolation='none', aspect='auto')
    ax_ss.set_xlim(-20, 20)
    ax_ss.set_ylim(0, 20)
    ax_ss.set_xlabel(fd.unit.to_string('latex'))
    ax_ss.set_ylabel(tau.unit.to_string('latex'))

    plt.show()
    plt.close()

    # JODRELL BANK
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
    plot_screen(ax, jodrell, 0*u.kpc, color='blue')
    plot_screen(ax, s2, d2, color='orange')
    plot_screen(ax, pulsar, dp, color='green')

    obs_jb = jodrell.observe(s2.observe(pulsar, distance=dp-d2), distance=d2)
    path_shape = obs_jb.tau.shape  # Also trigger calculation of pos, vel.
    tpos = obs_jb.pos
    scat1 = obs_jb.source.pos
    ppos = obs_jb.source.source.pos
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
        ax.scatter(_x[1:3], _y[1:3], _z[1:3], marker='o',
                   color=['red', 'orange'])

    # Create dynamic spectrum using delay for each path.
    tau0 = np.hstack([obs_jb.tau.ravel()])
    taudot = np.hstack([obs_jb.taudot.ravel()])
    brightness = np.hstack([np.broadcast_to(obs_jb.brightness, obs_jb.tau.shape).ravel()])
    t = np.linspace(0, 120*u.min, 300)[:, np.newaxis]
    f = np.linspace(300*u.MHz, 310*u.MHz, 600)
    tau = (tau0[:, np.newaxis, np.newaxis]
           + taudot[:, np.newaxis, np.newaxis] * t)
    ph = phasor(f, tau)
    dw = ph * brightness[:, np.newaxis, np.newaxis]
    # Calculate and show dynamic spectrum.
    #ds = np.abs(dw.sum(0))**2
    #jb_ds = (np.abs(dw.sum(0))**2).T
    jb_ds = (dw.sum(0)).T
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
    ax_ss.imshow(np.log10(np.abs(ss.T)**2), vmin=-7, vmax=5, cmap='Greys',
                 extent=axis_extent(fd) + axis_extent(tau),
                 origin='lower', interpolation='none', aspect='auto')
    ax_ss.set_xlim(-20, 20)
    ax_ss.set_ylim(0, 20)
    ax_ss.set_xlabel(fd.unit.to_string('latex'))
    ax_ss.set_ylabel(tau.unit.to_string('latex'))

    #plt.show()
    plt.close()

    # VLA
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
    plot_screen(ax, jodrell, 0*u.kpc, color='blue')
    plot_screen(ax, s2, d2, color='orange')
    plot_screen(ax, pulsar, dp, color='green')

    obs_vla = vla.observe(s2.observe(pulsar, distance=dp-d2), distance=d2)
    path_shape = obs_vla.tau.shape  # Also trigger calculation of pos, vel.
    tpos = obs_vla.pos
    scat1 = obs_vla.source.pos
    ppos = obs_vla.source.source.pos
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
        ax.scatter(_x[1:3], _y[1:3], _z[1:3], marker='o',
                   color=['red', 'orange'])

    # Create dynamic spectrum using delay for each path.
    tau0 = np.hstack([obs_vla.tau.ravel()])
    taudot = np.hstack([obs_vla.taudot.ravel()])
    brightness = np.hstack([np.broadcast_to(obs_vla.brightness, obs_vla.tau.shape).ravel()])
    t = np.linspace(0, 120*u.min, 300)[:, np.newaxis]
    f = np.linspace(300*u.MHz, 310*u.MHz, 600)
    tau = (tau0[:, np.newaxis, np.newaxis]
           + taudot[:, np.newaxis, np.newaxis] * t)
    ph = phasor(f, tau)
    dw = ph * brightness[:, np.newaxis, np.newaxis]
    # Calculate and show dynamic spectrum.
    #ds = np.abs(dw.sum(0))**2
    #jb_ds = (np.abs(dw.sum(0))**2).T
    vla_ds = (dw.sum(0)).T
    ax_ds = plt.subplot(233)
    ax_ds.imshow((np.abs(dw.sum(0))**2).T, cmap='Greys',
                 extent=axis_extent(t) + axis_extent(f),
                 origin='lower', interpolation='none', aspect='auto')
    ax_ds.set_xlabel(t.unit.to_string('latex'))
    ax_ds.set_ylabel(f.unit.to_string('latex'))
    # And the conjugate spectrum.
    ss = np.fft.fft2(vla_ds)
    ss /= ss[0, 0]
    ss = np.fft.fftshift(ss)
    tau = np.fft.fftshift(np.fft.fftfreq(f.size, f[1]-f[0])).to(u.us)
    fd = np.fft.fftshift(np.fft.fftfreq(t.size, t[1]-t[0])).to(u.mHz)
    ax_ss = plt.subplot(236)
    ax_ss.imshow(np.log10(np.abs(ss.T)**2), vmin=-7, vmax=5, cmap='Greys',
                 extent=axis_extent(fd) + axis_extent(tau),
                 origin='lower', interpolation='none', aspect='auto')
    ax_ss.set_xlim(-20, 20)
    ax_ss.set_ylim(0, 20)
    ax_ss.set_xlabel(fd.unit.to_string('latex'))
    ax_ss.set_ylabel(tau.unit.to_string('latex'))

    #plt.show()
    plt.close()


    # MAKE CROSS SPECTRUM
    hamm_win = np.outer(np.hamming(len(tau)), np.hamming(len(fd)))
    ar_ds *= hamm_win
    jb_ds *= hamm_win
    vla_ds *= hamm_win
    ar_ss = np.fft.fft2(np.abs(ar_ds)**2)
    jb_ss = np.fft.fft2(np.abs(jb_ds)**2)
    vla_ss=np.fft.fft2(np.abs(vla_ds)**2)
    ar_ss /= ar_ss[0,0]
    jb_ss /= jb_ss[0,0]
    vla_ss /= vla_ss[0,0]
    ar_ss =np.fft.fftshift(ar_ss)
    jb_ss =np.fft.fftshift(jb_ss)
    vla_ss =np.fft.fftshift(vla_ss)
    cross_arjb = ar_ss * np.conj(jb_ss)
    cross_arvla = ar_ss * np.conj(vla_ss)
    delay_average_arjb = np.angle(cross_arjb[cross_arjb.shape[0]//2:].mean(0))
    delay_average_arvla = np.angle(cross_arvla[cross_arvla.shape[0]//2:].mean(0))

    fig = plt.figure(figsize=(20, 6))
    ax_ss = plt.subplot(231)
    plt.imshow(np.log10(np.abs(ss)**2), vmin=-7, vmax=5, cmap='Greys',
                 extent=axis_extent(fd) + axis_extent(tau),
                 origin='lower', interpolation='none', aspect='auto')
    plt.xlim(-10, 10)
    plt.ylim(0, 15)
    plt.xlabel(fd.unit.to_string('latex'))
    plt.ylabel(tau.unit.to_string('latex'))

    fig.add_subplot(232)
    plt.imshow(np.log10(np.abs(cross_arjb)), aspect='auto', interpolation='none', origin='lower',
               extent=[fd[0].value, fd[-1].value, tau[0].value, tau[-1].value])
    plt.plot(fd.value, fd.value**2 * eta.value, color='blue')
    plt.xlabel("Doppler Frequency [mHz]")
    plt.ylabel("Delay [$\mu s$]")
    plt.xlim(-10, 10)
    plt.ylim(0, 15)
    plt.title("AR-JB")
    plt.colorbar()

    fig.add_subplot(233)
    plt.imshow(np.angle(cross_arjb), aspect='auto', interpolation='none', origin='lower', cmap="RdBu",
               extent=[fd[0].value, fd[-1].value, tau[0].value, tau[-1].value],
               vmin=-np.pi/4, vmax=np.pi/4)
    plt.xlabel("Doppler Frequency [mHz]")
    plt.ylabel("Delay [$\mu s$]")
    plt.xlim(-10, 10)
    plt.ylim(0, 15)
    plt.title("$\\eta = {0:.5f}$".format(eta))
    plt.colorbar()


    fig.add_subplot(212)
    plt.plot(fd, delay_average_arjb, label="Delay Averaged Cross AR-JB")
    plt.plot(fd, fd * delay_jb * (2*np.pi) / 1e3, label = "Delay = {0:.3f}".format(delay_jb), color='red', linestyle='dotted')
    plt.xlim(-10, 10)
    plt.ylim(-1, 1)
    plt.xlabel("Doppler Frequency [mHz]")
    plt.ylabel("Radians")
    plt.legend()
    plt.ylim(-np.pi, np.pi)
    plt.tight_layout()
    plt.show()

    # ===

    fig = plt.figure(figsize=(20, 6))
    ax_ss = plt.subplot(231)
    plt.imshow(np.log10(np.abs(ss)**2), vmin=-7, vmax=5, cmap='Greys',
                 extent=axis_extent(fd) + axis_extent(tau),
                 origin='lower', interpolation='none', aspect='auto')
    plt.xlim(-10, 10)
    plt.ylim(0, 15)
    plt.xlabel(fd.unit.to_string('latex'))
    plt.ylabel(tau.unit.to_string('latex'))

    fig.add_subplot(232)
    plt.imshow(np.log10(np.abs(cross_arvla)), aspect='auto', interpolation='none', origin='lower',
               extent=[fd[0].value, fd[-1].value, tau[0].value, tau[-1].value])
    plt.plot(fd.value, fd.value**2 * eta.value, color='blue')
    plt.xlabel("Doppler Frequency [mHz]")
    plt.ylabel("Delay [$\mu s$]")
    plt.xlim(-10, 10)
    plt.ylim(0, 15)
    plt.title("AR-VLA")
    plt.colorbar()

    fig.add_subplot(233)
    plt.imshow(np.angle(cross_arvla), aspect='auto', interpolation='none', origin='lower', cmap="RdBu",
               extent=[fd[0].value, fd[-1].value, tau[0].value, tau[-1].value],
               vmin=-np.pi/4, vmax=np.pi/4)
    plt.xlabel("Doppler Frequency [mHz]")
    plt.ylabel("Delay [$\mu s$]")
    plt.xlim(-10, 10)
    plt.ylim(0, 15)
    plt.title("$\\eta = {0:.5f}$".format(eta))
    plt.colorbar()


    fig.add_subplot(212)
    plt.plot(fd, delay_average_arvla, label="Delay Averaged Cross AR-VLA")
    plt.plot(fd, fd * delay_vla * (2*np.pi) / 1e3, label = "Delay = {0:.3f}".format(delay_vla), color='red', linestyle='dotted')
    plt.xlim(-10, 10)
    plt.ylim(-1, 1)
    plt.xlabel("Doppler Frequency [mHz]")
    plt.ylabel("Radians")
    plt.legend()
    plt.ylim(-np.pi, np.pi)
    plt.tight_layout()
    plt.show()
