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

dp = 0.372*u.kpc
d2 = dp/2
d1 = dp/4


pulsar = Source(CartesianRepresentation([0., 0., 0.]*u.AU),
                vel=CartesianRepresentation(600, 0., 0., unit=u.km/u.s))


# ON JANUARY 27, 2019 the observatory velocities are given by:
# ARECIBO: <[ 18.44724718, -13.00367214] km / s (22.56981209 km / s, -125.18043988 deg)
# JODRELL: <[ 18.75964149, -13.07485774] km / s (22.8664832  km / s, -124.87523808 deg)
# VLA    : <[ 18.60568632, -12.93659457] km / s (22.66113507 km / s, -124.81101557 deg)

arecibo = Telescope(CartesianRepresentation([0., 0., 0.]*u.AU),
        vel=CylindricalRepresentation(22.56981209, (-125.18  - 11.4)*u.deg, 0.).to_cartesian()*u.km/u.s)

jodrell = Telescope(CylindricalRepresentation(5552,  (-42.33 - 11.4)*u.deg, 0.).to_cartesian() * u.km,
        vel=CylindricalRepresentation(22.8664832,  (-124.88  - 11.4)*u.deg, 0.).to_cartesian()*u.km/u.s)

vla     = Telescope(CylindricalRepresentation(3937,  (57.364 - 11.4)*u.deg, 0.).to_cartesian() * u.km,
        vel=CylindricalRepresentation(22.8664832,  (-124.88  - 11.4)*u.deg, 0.).to_cartesian()*u.km/u.s)

screen1 = Screen1D(CylindricalRepresentation(1., 45*u.deg, 0.).to_cartesian(),
            np.array([0.00001])*u.AU,
            magnification=np.array([1]))

screen2 = Screen1D(CylindricalRepresentation(1., (63.6 - 11.4)*u.deg, 0.).to_cartesian(),
              0.5*np.array([-0.711, -0.62, -0.53, -0.304, -0.111, -0.052, -0.031, 0.0001, 0.0201, 0.0514, 0.102, 0.199, 0.3001, 0.409])*u.AU,
              magnification=np.array([0.01, 0.01, 0.02, 0.08, 0.25j, 0.34, 0.4+.1j,1, 0.2-.5j, 0.5j, 0.3, 0.2, 0.09, 0.02]))

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
    plot_screen(ax, arecibo, 0*u.kpc, color='blue')
    plot_screen(ax, screen1, d1, color='red')
    plot_screen(ax, screen2, d2, color='orange')
    plot_screen(ax, pulsar, dp, color='green')
    # Connect origins
    # ax.plot(np.zeros(4), np.zeros(4),
    #         [0., d1.value, d2.value, dp.value], color='black')


    # ARECIBO
    obscreen2 = arecibo.observe(
        screen1.observe(screen2.observe(pulsar, distance=dp-d2),
            distance=d2-d1), distance=d1)
    path_shape = obscreen2.tau.shape  # Also trigger calculation of pos, vel.
    tpos = obscreen2.pos
    scat1 = obscreen2.source.pos
    scat2 = obscreen2.source.source.pos
    ppos = obscreen2.source.source.source.pos
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
    tau0 = np.hstack([obscreen2.tau.ravel()])
    taudot = np.hstack([obscreen2.taudot.ravel()])
    brightness = np.hstack([np.broadcast_to(obscreen2.brightness, obscreen2.tau.shape).ravel()])
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
    ax_ss.imshow(np.log10(np.abs(ss.T)**2), vmin=-7, vmax=0, cmap='Greys',
                 extent=axis_extent(fd) + axis_extent(tau),
                 origin='lower', interpolation='none', aspect='auto')
    ax_ss.set_xlim(-20, 20)
    ax_ss.set_ylim(0, 20)
    ax_ss.set_xlabel(fd.unit.to_string('latex'))
    ax_ss.set_ylabel(tau.unit.to_string('latex'))

    #plt.show()
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
    plot_screen(ax, arecibo, 0*u.kpc, color='blue')
    plot_screen(ax, screen1, d1, color='red')
    plot_screen(ax, screen2, d2, color='orange')
    plot_screen(ax, pulsar, dp, color='green')
    obs3 = jodrell.observe(
        screen1.observe(screen2.observe(pulsar, distance=dp-d2),
            distance=d2-d1), distance=d1)
    path_shape = obs3.tau.shape  # Also trigger calculation of pos, vel.
    tpos = obs3.pos
    scat1 = obs3.source.pos
    scat2 = obs3.source.source.pos
    ppos = obs3.source.source.source.pos
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
    tau0 = np.hstack([obs3.tau.ravel()])
    taudot = np.hstack([obs3.taudot.ravel()])
    brightness = np.hstack([np.broadcast_to(obs3.brightness, obs3.tau.shape).ravel()])
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
    ax_ss.imshow(np.log10(np.abs(ss.T)**2), vmin=-7, vmax=0, cmap='Greys',
                 extent=axis_extent(fd) + axis_extent(tau),
                 origin='lower', interpolation='none', aspect='auto')
    ax_ss.set_xlim(-20, 20)
    ax_ss.set_ylim(0, 20)
    ax_ss.set_xlabel(fd.unit.to_string('latex'))
    ax_ss.set_ylabel(tau.unit.to_string('latex'))

    #plt.show()
    plt.close()



    # MAKE CROSS SPECTRUM

    # Apply window functions
    hamm_win = np.outer(np.hamming(len(tau)), np.hamming(len(fd)))
    ar_ds *= hamm_win
    jb_ds *= hamm_win


    ar_ss = np.fft.fftshift(np.fft.fft2(ar_ds))
    jb_ss = np.fft.fftshift(np.fft.fft2(jb_ds))
    cross = ar_ss * np.conj(jb_ss)

    # Calculate the curvature (quickly)
    cross_copy = cross.copy()
    fd_mincut = np.searchsorted(fd.value, -1.2)
    fd_maxcut = np.searchsorted(fd.value, 1.2)
    tau_min = np.searchsorted(tau.value, 0.5)
    cross_copy[:,fd_mincut:fd_maxcut] = 0
    cross_copy[:tau_min,] = 0
    x_max, y_max = np.unravel_index((np.abs(cross_copy)**2).argmax(), cross_copy.shape)
    #print(fd[y_max], tau[x_max])
    curvature = (tau[x_max] / (fd[y_max])**2).to(u.s**3)
    print("Curvature = {0:.5f}".format(curvature))

    # Compute the curvature one would observe is just single screen at each location
    s_1 = 1 - (d1/dp).to(u.dimensionless_unscaled)
    s_2 = 1 - (d2/dp).to(u.dimensionless_unscaled)
    s_3 = 1 - ((d2-d1)/(dp-d1)).to(u.dimensionless_unscaled)

    deff1 = dp * (1-s_1)/s_1
    deff2 = dp * (1-s_2)/s_2
    deff3 = (dp-d1) * (1-s_3)/s_3
    veff1 = np.dot(pulsar.vel.get_xyz() * (1 - s_1)/s_1 + arecibo.vel.get_xyz(), screen1.normal.get_xyz())
    veff2 = np.dot(pulsar.vel.get_xyz() * (1 - s_2)/s_2 + arecibo.vel.get_xyz(), screen2.normal.get_xyz())
    veff3 = np.dot(pulsar.vel.get_xyz() * (1 - s_3)/s_3, screen2.normal.get_xyz())
    lambd = ac.c / (305 * u.MHz)
    eta1 = ((lambd**2/(2*ac.c)) * deff1/(veff1**2)).to(u.s**3)
    eta2 = ((lambd**2/(2*ac.c)) * deff2/(veff2**2)).to(u.s**3)
    eta3 = ((lambd**2/(2*ac.c)) * deff3/(veff3**2)).to(u.s**3)
    print("eta1 = {:.5f}".format(eta1))
    print("eta2 = {:.5f}".format(eta2))
    print("eta3 = {:.5f}".format(eta3))


    fig = plt.figure(figsize=(12, 6))
    fig.add_subplot(221)
    plt.imshow(np.log10(np.abs(cross)), aspect='auto', interpolation='none', origin='lower',
               extent=[fd[0].value, fd[-1].value, tau[0].value, tau[-1].value])
    plt.plot(fd.value, fd.value**2 * curvature, color='red')
    plt.plot(fd.value, fd.value**2 * eta3, color='blue')
    plt.xlabel("Doppler Frequency [mHz]")
    plt.ylabel("Delay [$\mu s$]")
    plt.xlim(-10, 10)
    plt.ylim(0, 10)
    plt.title("AR-YY")
    plt.colorbar()

    fig.add_subplot(222)
    plt.imshow(np.angle(cross), aspect='auto', interpolation='none', origin='lower', cmap="RdBu",
               extent=[fd[0].value, fd[-1].value, tau[0].value, tau[-1].value],
               vmin=-np.pi/2, vmax=np.pi/2)
    plt.xlabel("Doppler Frequency [mHz]")
    plt.ylabel("Delay [$\mu s$]")
    plt.xlim(-10, 10)
    plt.ylim(0, 10)
    plt.title("$\\eta = {0:.5f}$".format(curvature))
    plt.colorbar()



    # Average along Delay Axes
    delay_average = np.angle(cross[cross.shape[0]//2:].mean(0))

    def f(x, a):
        return a*x

    s = curve_fit(f, fd[145:155], delay_average[145:155])
    slope = s[0][0]
    delay = slope * 1000 / (2 * np.pi)
    print("Delay = {:.5f}".format(delay))


    fig.add_subplot(223)
    plt.plot(delay_average, label="Delay Averaged Cross")
    plt.xlabel("Doppler Frequency Bins")
    plt.ylabel("Radians")
    plt.legend()
    plt.ylim(-0.25, 0.25)
    plt.tight_layout()


    fig.add_subplot(224)
    plt.plot(fd, delay_average, label="Delay Averaged Cross")
    plt.plot(fd, fd * slope, label = "Delay = {0:.3f} s".format(delay))
    plt.xlabel("Doppler Frequency [mHz]")
    plt.ylabel("Radians")
    plt.legend()
    plt.ylim(-0.25, 0.25)
    plt.tight_layout()

    plt.show()
    plt.close()
