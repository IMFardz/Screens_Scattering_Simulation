# Licensed under the GPLv3 - see LICENSE
"""Simulate scintillation through two screens.
The setup is somewhat similar to what is seen in the Brisken data,
with one screen that has a lot of images, and another that has only
one.
The geometry of the paths is shown, as well as inferred dynamic and
secondary spectra.
.. warning:: Usage quite likely to change.

In this altered version I change it so that the screens are VERY close
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
import astropy.units as u


dp = 0.372*u.kpc
d2 = 0.110*u.kpc
d1 = 0.090*u.kpc


pulsar = Source(CartesianRepresentation([0., 0., 0.]*u.AU),
                vel=CartesianRepresentation(655.848, 0., 0., unit=u.km/u.s))
arecibo = Telescope(CartesianRepresentation([0., 0., 0.]*u.AU))
jodrell = Telescope(CylindricalRepresentation(5552, (-42.33 - 11.4)*u.deg, 0.).to_cartesian() * u.km)

s1 = Screen1D(CylindricalRepresentation(1., (63.6 - 11.4)*u.deg, 0.).to_cartesian(),
     np.array([-0.711, -0.62, -0.53, -0.304, -0.111, -0.052, -0.031, 0.0001, 0.0201, 0.0514, 0.102, 0.199, 0.3001, 0.409])*u.AU,
     v = 0 * np.ones(14) * u.km/u.s,
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
    plot_screen(ax, s1, d1, color='red')
    plot_screen(ax, pulsar, dp, color='green')
    plt.show()
