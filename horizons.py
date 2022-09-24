"""Cosmological horizons."""

import numpy as np
import matplotlib.pyplot as plt
import astropy.cosmology as acosmo
import scipy.integrate
import pandas as pd
from numba import njit
from matplotlib import animation
import matplotlib.cm as cm
from scipy.optimize import fsolve
from scipy.integrate._ivp.rk import OdeSolver
from tqdm import tqdm

__M_2_LY__ = 1.06e-16

__S_2_Y__ = 3.17e-8

__G__ = 6.6e-11 # m**3 / s**2 / kg

# MONKEY PATCHING FROM :
# https://towardsdatascience.com/do-stuff-at-each-ode-integration-step-monkey-patching-solve-ivp-359b39d5f2

# monkey patching the ode solvers with a progress bar #
# save the old methods - we still need them
old_init = OdeSolver.__init__
old_step = OdeSolver.step


# define our own methods
def new_init(self, fun, t0, y0, t_bound, vectorized, support_complex=False):

    # define the progress bar
    self.pbar = tqdm(total=t_bound - t0, unit='Gyr', initial=t0, desc='IVP')
    self.last_t = t0

    # call the old method - we still want to do the old things too!
    old_init(self, fun, t0, y0, t_bound, vectorized, support_complex)


def new_step(self):
    # call the old method
    old_step(self)

    # update the bar
    tst = self.t - self.last_t
    self.pbar.update(tst)
    self.last_t = self.t

    # close the bar if the end is reached
    if self.t >= self.t_bound:
        self.pbar.close()


# overwrite the old methods with our customized ones
OdeSolver.__init__ = new_init
OdeSolver.step = new_step

# END OF MONKEY PATCHING #


@njit(cache=True)
def interp_nb(x_vals, x, y):
    """Numba interpolation.

    Parameters
    ----------
    x_vals : float
        Value to interpolate.
    x : numpy.array(float)
        x values array.
    y : numpy.array(float)
         y = f(x) values array.

    Returns
    -------
    float
        f(x_vals).

    """
    return np.interp(x_vals, x, y)


@njit(cache=True)
def compute_grav(r, ang):
    x = r * np.cos(ang)
    y = r * np.sin(ang)
    Gm = __G__ * 2e42 * __M_2_LY__**3/__S_2_Y__**2
    Ax = np.zeros(len(x))
    Ay = np.zeros(len(y))

    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                N = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                Ax[i] += -Gm * (x[i] - x[j]) / N**3
                Ay[i] += -Gm * (y[i] - y[j]) / N**3
    Ax += -100 * Gm * x / np.abs(x**2 + y**2)**3
    Ay += -100 * Gm * y / np.abs(x**2 + y**2)**3
    return Ax, Ay


@njit(cache=True)
def update_r(r, ang, H_t, dt, Vx, Vy):
    r"""Update proper distance of non-moving object.

    Parameters
    ----------
    r : numpy.array(float)
        Objects proper distances.
    ang : numpy.array(float)
        Ojects angular positions.
    H_t : float
        Hubble parameter H(t) in Gyr$^-1$.
    dt : float
        Time step in Gyr.

    Returns
    -------
    numpy.array(float)
        New position after expansion.

    Notes
    -----
    The evolution is given by :
    ..maths:
        \frac{dr}{dt} = H(t) r
    """

    new_r = r + H_t * r * dt
    xp, yp = new_r * np.cos(ang) + Vx * dt, new_r * np.sin(ang) + Vy * dt
    Ax, Ay = compute_grav(r, ang)
    Vx += Ax * dt
    Vy += Ay * dt
    return new_r, ang, xp, yp, Vx, Vy


@njit(cache=True)
def update_rw(r, ang, H_t, dt):
    r"""Update photons positions.

    Parameters
    ----------
    r : numpy.array(float)
        Photons proper distances.
    ang : type
        Photons angular positions.
    H_t : float
        Hubble parameter H(t) in Gyr$^-1$.
    dt : float
        Time step in Gyr.

    Returns
    -------
    numpy.array(float)
        New photons positions.

    Notes
    -----
    The evolution is given by :
    ..maths:
        \frac{dr}{dt} = H(t) r - c
    """
    new_r = r + r * H_t * dt - dt
    new_r, ang = new_r[new_r > 0], ang[new_r > 0]
    xp, yp = new_r * np.cos(ang), new_r * np.sin(ang)
    return new_r, ang, xp, yp


def set_cosmo(cosmo):
    """Load an astropy cosmological model.

    Parameters
    ----------
    cosmo : dict
        A dict containing cosmology parameters.

    Returns
    -------
    astropy.cosmology.object
        An astropy cosmological model.

    """
    astropy_mod = list(map(lambda x: x.lower(), acosmo.parameters.available))
    if isinstance(cosmo, str):
        name = cosmo.lower()
        if name in astropy_mod:
            if name == 'planck18':
                return acosmo.Planck18
            elif name == 'planck18_arxiv_v2':
                return acosmo.Planck18_arXiv_v2
            elif name == 'planck15':
                return acosmo.Planck15
            elif name == 'planck13':
                return acosmo.Planck13
            elif name == 'wmap9':
                return acosmo.WMAP9
            elif name == 'wmap7':
                return acosmo.WMAP7
            elif name == 'wmap5':
                return acosmo.WMAP5
        else:
            raise ValueError(f'Available model are {astropy_mod}')
    elif isinstance(cosmo, dict):
        if 'Ode0' not in cosmo.keys():
            if 'Ok0' in cosmo.keys():
                Ok0 = cosmo['Ok0']
                cosmo.pop('Ok0')
            else:
                Ok0 = 0.
            cosmo['Ode0'] = 1 - cosmo['Om0'] - Ok0
        return acosmo.w0waCDM(**cosmo)
    else:
        return cosmo


class CosmoHorizon:
    """Class to compute cosmological horizons.

    Parameters
    ----------
    cosmo : dict/str/astropy.cosmology.FLRW
        Cosmological model.
    file : str
        File that contains already computed cosmology.

    Attributes
    ----------
    t_amax : float
        Time when $\frac{da}{dt} = 0$.
    cosmo_tab : pandas.DataFrame
        Computed data.
    t_today : float
        Time when a = 1.
    cosmo

    """

    def __init__(self, cosmo=None, file=None):
        """Init the CosmoHorizon class."""
        if cosmo is not None:
            self.cosmo = set_cosmo(cosmo)
            if self.cosmo.Ok0 < 0:
                a = self.cosmo.Om0 / (self.cosmo.Om0 - 1)
                z_amax = fsolve(self.cosmo.H, 1 / a - 1)
                self.t_amax = self.cosmo.age(z_amax)[0].value
            else:
                self.t_amax = np.inf
        else:
            self.cosmo = None
        if file is not None:
            self.cosmo_tab = pd.read_csv(file, index_col=0)
            self.t_today = interp_nb(1,
                                     self.cosmo_tab['a'].to_numpy(),
                                     self.cosmo_tab['time'].to_numpy())
        else:
            self.cosmo_tab = None

    def _da_dt(self, t, a):
        """Scale factor differential equation.

        Parameters
        ----------
        t : float
            Time in Gyr.
        a : float
            Scale factor.

        Returns
        -------
        float
            $\frac{da}{dt}$ at time t.

        """
        z = 1/a - 1
        da = a * self.cosmo.H(z).to('Gyr-1').value
        if t > self.t_amax:
            da *= -1
        return da

    def compute_horizon(self, output_file, max_t=1e2, a0=1e-6, max_step=100):
        """Compute the cosmology in funtio of time.

        Parameters
        ----------
        output_file : str
            A file to store results.
        a0 : float
            Scale factor taking as starting point.
        max_step : float
            Maximum time step between two points in Gyr.

        Returns
        -------
        None
            Write a csv file and set self.cosmo_tab.

        """
        if self.cosmo is None:
            print('No cosmo, no computation')
            return
        res = scipy.integrate.solve_ivp(self._da_dt, [0, max_t + 1],
                                        y0=[a0],
                                        max_step=max_step)
        time = res['t']
        scale_factor = res['y'][0]
        dscale = (scale_factor[1:] - scale_factor[:-1]) / (time[1:] - time[:-1])
        part_h = scale_factor[:-1] * np.cumsum(1 / scale_factor[:-1] * (time[1:] - time[:-1]))
        evt_h = self._compute_event_h(time, scale_factor)
        H = dscale / scale_factor[:-1]

        dic = {'time': time[:-1],
               'a': scale_factor[:-1],
               'dadt': dscale,
               'H': H,
               'part_h': part_h,
               'evt_h': evt_h[:-1],
               'H_h': 1 / H
               }

        self.cosmo_tab = pd.DataFrame(dic)
        self.cosmo_tab.to_csv(output_file)
        mask = self.cosmo_tab['time'] < self.t_amax
        self.t_today = interp_nb(1,
                                 self.cosmo_tab['a'][mask].to_numpy(),
                                 self.cosmo_tab['time'][mask].to_numpy())

    def _compute_event_h(self, time, scale_factor):
        """Compute the event horizon.

        Parameters
        ----------
        time : numpy.array(float)
            Times at which compute the event horizon.
        scale_factor : numpy.array(float)
            a(t).

        Returns
        -------
        numpy.array(float)
            Cosmological event horizon.

        """
        def H_inv(z):
            return 1 / self.cosmo.H(z).to('Gyr-1').value
        evt_h = scale_factor * np.array([np.trapz(1 / scale_factor[i:], x=time[i:])
                                        for i in range(len(time))])
        last_z = 1 / scale_factor[-1] - 1
        evt_h += scale_factor * scipy.integrate.quad(H_inv, -1 + 1e-20, last_z)[0]
        return evt_h

    def plot(self, d1, d2, xlim=None, ylim=None):
        """Plot data x = d1, y = d2.

        Parameters
        ----------
        d1 : str
            x axis parameter.
        d2 : type
            y axis parameter.
        xlim : type
            plot xlim.
        ylim : type
            plot ylim.

        Returns
        -------
        None
            Just plot data.

        Notes
        -----
        Available data are :
        'time', 'H', 'dadt', 'H_h', 'evt_h', 'part_h'

        """
        plot_dic = {'time': 'Time in Gyr',
                    'H': 'Hubble parameter H in Gly',
                    'a': 'Scale factor a',
                    'dadt': '$\dot{a}$',
                    'H_h': 'Hubble horizon $L_H$ in Gly',
                    'evt_h': 'Event Horizon $L_p$ in Gly',
                    'part_h': 'Particle Horizon $L_p$ in Gly'
                    }

        plt.figure(dpi=200)

        plt.xlabel(plot_dic[d1])
        plt.ylabel(plot_dic[d2])
        plt.plot(self.cosmo_tab[d1], self.cosmo_tab[d2])
        plt.axvline(interp_nb(self.t_today,
                              self.cosmo_tab['time'].to_numpy(),
                              self.cosmo_tab[d1].to_numpy()), c='k', ls='--')

        plt.axhline(interp_nb(self.t_today,
                              self.cosmo_tab['time'].to_numpy(),
                              self.cosmo_tab[d2].to_numpy()), c='k', ls='--')
        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])
        plt.show()

    def sim_anim(self, t_range, nframes, nparts=1, gen_lim=[0.1, 1.5], lim=20, norm=None, **kwargs):
        """Create an animation.

        Parameters
        ----------
        t_range : list(float, float)
            Time range of the animation in Gyr.
        nframes : int
            Number of images frames.
        nparts : int
            Number of particles.
        gen_lim : list(float, float)
            Range of particles generation in Glyr.
        lim : float
            Plot limit.
        norm : str, opt
            'H_h' or 'evt_h' -> normalize coordinates.
        **kwargs : type
            AnimHorizon kwargs.

        Returns
        -------
        type
            Description of returned object.

        """
        self.anim = AnimHorizon(self.cosmo_tab,
                                t_range,
                                nframes,
                                nparts=nparts,
                                gen_lim=gen_lim,
                                lim=lim,
                                norm=norm)
        self.anim.animate(**kwargs)


class AnimHorizon(object):
    def __init__(self, df, t_range, nframes, nparts=0, gen_lim=[0.1, 1.5], lim=20, norm=None):
        self.df = df
        self.t_range = t_range
        self.nframes = nframes
        self.norm = norm
        self.nparts = nparts
        self.gen_lim = gen_lim
        self.dt = (self.t_range[1] - self.t_range[0]) / nframes
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-lim, lim), ylim=(-lim, lim))
        self.fig.gca().set_aspect('equal')
        self.init_plotobj()

    def init_plotobj(self):
        self.obs, = self.ax.plot([0], [0], 'o', c='k', ms=3)

        self.hubble_h_plot, = self.ax.plot([], [], label='Hubble Horizon')

        self.evt_h_plot, = self.ax.plot([], [], c='k', ls='--', label='Event Horizon')

        self.parts, = self.ax.plot([], [], 'o', c='crimson', ms=2)

        self.photons = self.ax.scatter([], [], s=1)

    def _ani_init(self):
        theta = np.linspace(0, 2 * np.pi, 500)
        self.circle = (np.cos(theta), np.sin(theta))
        self.evt_h = interp_nb(self.t_range[0],
                               self.df['time'].to_numpy(),
                               self.df['evt_h'].to_numpy())
        self.part_h = interp_nb(self.t_range[0],
                                self.df['time'].to_numpy(),
                                self.df['part_h'].to_numpy())

        self.H = np.interp(self.t_range[0],
                           self.df['time'].to_numpy(),
                           self.df['H'].to_numpy())

        self.a_em = interp_nb(self.t_range[0],
                              self.df['time'].to_numpy(),
                              self.df['a'].to_numpy())
        self.a_t = np.copy(self.a_em)

        self.cmap = cm.rainbow.reversed()

        r0 = np.random.uniform(self.gen_lim[0], self.gen_lim[1], size=self.nparts)
        ang0 = np.random.uniform(0, 2 * np.pi, size=self.nparts)
        V0 = np.random.normal(0, 9.4e-4, size=self.nparts)
        angV0 = np.random.uniform(0, 2 * np.pi, size=self.nparts)
        Vx0, Vy0 = V0 * np.cos(angV0), V0 * np.sin(angV0)

        self.parts_coords = [r0, ang0, r0 * np.cos(ang0), r0 * np.sin(ang0), Vx0, Vy0]
        self.photons_coords = [r0, ang0, r0 * np.cos(ang0), r0 * np.sin(ang0),  np.ones(len(r0))]

        self.obs, = self.ax.plot([0], [0], 'o', c='k', ms=3)

        self.hubble_h_plot.set_data(self.H * self.circle[0] * self.norm_factor,
                                    self.H * self.circle[1] * self.norm_factor)

        self.evt_h_plot.set_data(self.evt_h * self.circle[0] * self.norm_factor,
                                 self.evt_h * self.circle[1] * self.norm_factor)

        self.parts.set_data(self.parts_coords[2] * self.norm_factor,
                            self.parts_coords[3] * self.norm_factor)

        self.photons.set_offsets(np.c_[self.photons_coords[2] * self.norm_factor,
                                       self.photons_coords[3] * self.norm_factor])

        self.photons.set_color(self.cmap(len(r0)))

        return self.obs, self.hubble_h_plot, self.evt_h_plot, self.parts, self.photons

    @property
    def norm_factor(self):
        if self.norm == 'H_h':
            nf = self.H
        elif self.norm == 'evt_h':
            nf = 1 / self.evt_h
        else:
            nf = 1
        return nf

    def ani_update(self, i):
        t = self.t_range[0] + i * self.dt
        self.evt_h = interp_nb(t,
                               self.df['time'].to_numpy(),
                               self.df['evt_h'].to_numpy())
        self.H = interp_nb(t,
                           self.df['time'].to_numpy(),
                           self.df['H'].to_numpy())
        self.a_t = interp_nb(t,
                             self.df['time'].to_numpy(),
                             self.df['a'].to_numpy())

        self.parts_coords = update_r(self.parts_coords[0],
                                     self.parts_coords[1],
                                     self.H,
                                     self.dt,
                                     self.parts_coords[-2],
                                     self.parts_coords[-1])

        self.photons_coords[:-1] = update_rw(self.photons_coords[0],
                                             self.photons_coords[1],
                                             self.H,
                                             self.dt)

        self.evt_h_plot.set_data(self.evt_h * self.circle[0] * self.norm_factor,
                                 self.evt_h * self.circle[1] * self.norm_factor)
        self.parts.set_data(self.parts_coords[2] * self.norm_factor,
                            self.parts_coords[3] * self.norm_factor)

        self.photons.set_offsets(np.c_[self.photons_coords[2] * self.norm_factor,
                                       self.photons_coords[3] * self.norm_factor])
        self.photons.set_color(self.cmap(self.a_em / self.a_t))

        self.hubble_h_plot.set_data(1 / self.H * self.circle[0] * self.norm_factor,
                                    1 / self.H * self.circle[1] * self.norm_factor)
        return self.hubble_h_plot, self.evt_h_plot, self.parts, self.photons

    def animate(self, interval=5, blit=False, repeat=False):
        self.anim = animation.FuncAnimation(self.fig,
                                            self.ani_update,
                                            init_func=self._ani_init,
                                            frames=self.nframes,
                                            interval=interval,
                                            blit=blit,
                                            repeat=repeat)
        plt.show()

    def save_anim(self, fname, fps=60, **kwargs):
        writermp4 = animation.FFMpegWriter(fps=fps)
        self.anim.save(fname, writer=writermp4, **kwargs)
