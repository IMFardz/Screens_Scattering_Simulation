"""
Simpler version of code used in diff_scint package
"""
import numpy as np
from astropy.time import Time, TimeDelta
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.constants as ac


# GLOBAL PARAMETERS. HAVE TO CHECK THESE LATER.

# The location of the pulsar:
pulsar_location = SkyCoord.from_name("PSR B1133+16")

# The observing wavelength
wavelength = ac.c / (332 * u.MHz)

# The distance of the pulsar in meters
d_psr = (0.37 * u.kpc).to(u.m)

# The velocity of the pulsar is m/s
vel_psr = (659.7 * u.km / u.s).to(u.m / u.s)

# PROPER MOTION of B1133+16 from PSRCAT
RA = -73.785 * u.mas/u.year
DEC = 366.569 * u.mas/u.year

# Get Velocity of Pulsar in UV Plane
u_psr = (RA.to(u.radian/u.second) * d_psr / u.rad).to(u.m/u.s)
v_psr = (DEC.to(u.radian/u.second)* d_psr / u.rad).to(u.m/u.s)
#print(u_psr, v_psr)
#print(np.sqrt(u_psr**2 + v_psr**2))
#print(np.arctan(v_psr / u_psr).to(u.deg) + 90*u.deg)

# The locations of the three stations, in ITRS coordinates at Julian time J2000
ar  = EarthLocation.from_geocentric(x=2390490.0,   y=-5564764.0,    z=1994727.0,    unit='meter')
jb  = EarthLocation.from_geocentric(x=3822626.04,  y=-154105.65,    z=5086486.04,   unit='meter')
vla = EarthLocation.from_geocentric(x=-1601192.,   y=-5041981.4,    z=3554871.4,    unit='meter')

# A dictionary of the stations
stations = {"AR": ar, "JB": jb, "YY": vla}


# ===============================================================================
# Make TIME ARRAYS

# The start time for all baselines
start_time = Time("2019-01-27 06:17:20.000", scale="utc")

# The size of each time bin
increment = TimeDelta(10*u.s, format='sec')

# The stop time for the AR x VLA baseline
ar_vla_stop_time = Time("2019-01-27 08:29:50.000", scale="utc")

# The stop time for the AR x JB baseline
ar_jb_stop_time = Time("2019-01-27 08:30:00.000", scale="utc")

# The stop time for the AR x VLA baseline
vla_jb_stop_time = Time("2019-01-27 08:29:50.000", scale="utc")

# Lists to store times
ar_vla_time = []
ar_jb_time  = []
vla_jb_time = []

# Fill up time arrays
current_time = start_time
while current_time < ar_vla_stop_time:
    ar_vla_time.append(current_time)
    current_time += increment

current_time = start_time
while current_time < ar_jb_stop_time:
    ar_jb_time.append(current_time)
    current_time += increment

current_time = start_time
while current_time < vla_jb_stop_time:
    vla_jb_time.append(current_time)
    current_time += increment

# Convert the types of the arrays
ar_vla_time = Time(ar_vla_time)
ar_jb_time  = Time(ar_jb_time)
vla_jb_time = Time(vla_jb_time)


# =====================================================================================================
# HELPER FUNCTIONS

def calculate_uvw(source, time, s1,s2):
    '''
    Calculates the projected uvw track for the baseline s2-s1.
    Negative u points West, Negative v points South.
    '''
    b = s2.get_gcrs(time).cartesian.xyz.to(u.m) - s1.get_gcrs(time).cartesian.xyz.to(u.m)
    H = -source.ra
    d     = source.dec
    trans = np.array([[ np.sin(H),           np.cos(H),                   0],
                      [-np.sin(d)*np.cos(H), np.sin(d)*np.sin(H),np.cos(d)],
                      [ np.cos(d)*np.cos(H),-np.cos(d)*np.sin(H),np.sin(d)]])
    return np.dot(trans,b)


def get_alpha_s(alpha0, alpha1, m0, m1, b0, b1):
    alpha_s = alpha0 + np.arctan((1 / np.sin(alpha1 - alpha0)) * (m1 / m0).value * (b0 / b1).value -
                                 (1 / np.tan(alpha1 - alpha0)))
    return alpha_s


def get_Veff(b, alpha, alpha_s, m):
    veff =  np.abs((b * np.cos(alpha - alpha_s))/m)
    return veff

def getDeff(Veff, eta, wavelength):
    deff= (2 * ac.c * Veff**2 * eta)/(wavelength**2)
    return deff

def getDistance(deff, d_psr):
    dist =  (deff)/(1 + (deff.to(u.kpc)/d_psr.to(u.kpc)).value)
    return dist

def getFractionalDistance(deff, d_psr):
    s =  1 / (1 + (deff.to(u.kpc)/d_psr.to(u.kpc)).value )
    return s

# =====================================================================================================
# MAIN FUNCTION

def calculate_screen_parameters(delay_aryy, delay_arjb, curvature):
    """Calculates the screen parameters"""
    # The location of the first baseline in (u,v) coordinates in units meters
    u_yy, v_yy, w_yy = calculate_uvw(pulsar_location, ar_vla_time, ar, vla)
    baseline0 = np.array([u_yy.to(u.km).value, v_yy.to(u.km).value, w_yy.to(u.km).value]) * u.km

    # The location of the second baseline in (u,v) coordinates in units meters
    u_jb, v_jb, w_jb = calculate_uvw(pulsar_location, ar_jb_time, ar, jb)
    baseline1 = np.array([u_jb.to(u.km).value, v_jb.to(u.km).value, w_jb.to(u.km).value]) * u.km

    # Calculate the constant alpha_yy, measured wrt. the v axis in clockwise direction.
    alpha_yy = np.pi/2 - np.arctan2(u_yy[len(u_yy)//2].value, v_yy[len(v_yy)//2].value)
    print("Alpha 0 = ", alpha_yy," radians")

    # Calculate the constant alpha_jb
    alpha_jb = np.pi/2 - np.arctan2(u_jb[len(u_jb)//2].value , v_jb[len(v_jb)//2].value)
    print("Alpha 1 = ", alpha_jb, " radians")

    # Calculate the length and error of the two baselines:
    b0 = np.sqrt(u_yy[len(u_yy)//2]**2 + v_yy[len(v_yy)//2]**2)
    b1 = np.sqrt(u_jb[len(u_jb)//2]**2 + v_jb[len(v_jb)//2]**2)

    # Calculate Parameters for each screen
    alpha_s  = get_alpha_s(alpha_yy, alpha_jb, delay_aryy, delay_arjb, b0, b1)
    veff     = get_Veff(b1, alpha_jb, alpha_s, delay_arjb)
    deff     = getDeff(veff, curvature, wavelength)
    distance = getDistance(deff, d_psr)
    s        = getFractionalDistance(deff, d_psr)


    # PRINT THE RESULTS:
    print("RESULTS FOR SCREEN:")
    #print("Curvature: ", curvature)
    print("Scattering Angle of Screen: {0}".format((alpha_s - np.pi/2) * 180 / np.pi))
    #print("Scattering Angle of Screen: {0}".format(alpha_s))
    print("Effective velocity: {0}".format(veff.to(u.km/u.s)))
    #print("Effective distance: {0}".format(deff.to(u.kpc)))
    #print("Distance to the Screen: {0}".format( distance.to(u.kpc)))
    print("s= {0}".format(s))

    return 0

if __name__=="__main__":
    # Test run on screen measuremennts used in diff_scint_B1133 package
    #calculate_screen_parameters(-26.56119*u.s, 10.31337*u.s, 0.27 * u.s**3)

    # # Second Screen 0.4 d_psr away:
    # calculate_screen_parameters(0.42734*u.s, 7.49501*u.s, 0.83237 * u.s**3)

    # # Second Screen 0.3 d_psr away:
    # calculate_screen_parameters(0.19530*u.s, 3.58213*u.s, 0.38593 *u.s**3)

    # # Second Screen 0.2 d_psr away:
    # calculate_screen_parameters(0.11877*u.s, 2.45653*u.s, 0.18607 *u.s**3)

    # # Second Screen 0.1 d_psr away:
    # calculate_screen_parameters(0.03589*u.s, 1.87401*u.s, 0.15949 *u.s**3)

    # ========
    # Varying Far SCREEN

    # # First Screen at 0.6 d_psr away:
    # calculate_screen_parameters(0.427*u.s, 7.495*u.s, 0.83237 * u.s**3)

    # # First Screen at 0.7 d_psr away:
    # calculate_screen_parameters(0.01952*u.s, 0.97692*u.s, 0.10817 * u.s**3)

    # # First Screen at 0.8 d_psr away:
    # calculate_screen_parameters(-0.00311*u.s, 0.642*u.s, 0.04624*u.s**3)

    # First Screen at 0.9 d_psr away:
    calculate_screen_parameters(0.02563*u.s, 0.261*u.s, 0.01541*u.s**3)
