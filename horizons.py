import numpy as np
import matplotlib.pyplot as plt
import astropy.cosmology as acosmo
import scipy.integrate
import pandas as pd
from numba import njit
from matplotlib import animation
import matplotlib.cm as cm
from scipy.optimize import fsolve


@njit(cache=True)
def compute_evt_h(time, scale_factor):
    evt_h = np.zeros(len(time)-1)
    for i in range(len(time[:-1])):
        evt_h[i] = scale_factor[i] * np.sum(1/scale_factor[i:-1] * (time[i+1:] - time[i:-1]))
    return evt_h


@njit(cache=True)
def interp_nb(x_vals, x, y):
    return np.interp(x_vals, x, y)


@njit(cache=True)
def update_r(r, ang, H_t, dt):
    new_r = r + H_t * r * dt
    xp, yp = new_r * np.cos(ang), new_r * np.sin(ang)
    return new_r, ang, xp, yp


@njit(cache=True)
def update_rw(r, ang, H_t, dt):
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
            cosmo['Ode0'] = 1 - cosmo['Om0']
        return acosmo.w0waCDM(**cosmo)
    else:
        return cosmo


class CosmoHorizon:
    def __init__(self, cosmo=None, file=None):
        if cosmo is not None:
            self.cosmo = set_cosmo(cosmo)
            if self.cosmo.Ok0 < 1:
                a = self.cosmo.Om0 / (self.cosmo.Om0 - 1)
                z_amax = fsolve(self.cosmo.H, 1/a - 1)
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
        z = 1/a - 1
        da = a * self.cosmo.H(z).to('Gyr-1').value
        if t > self.t_amax:
            da *= -1
        return da

    def compute_horizon(self, output_file, a0=1e-6, max_step=0.001):
        if self.cosmo is None:
            print('No cosmo, no computation')
            return
        res = scipy.integrate.solve_ivp(self._da_dt, [0, np.inf], y0=[a0], max_step=max_step)
        time = res['t']
        scale_factor = res['y'][0]
        dscale = (scale_factor[1:]-scale_factor[:-1])/(time[1:] - time[:-1])
        part_h = scale_factor[:-1] * np.cumsum(1/scale_factor[:-1] * (time[1:] - time[:-1]))
        evt_h = self.compute_event_h(time, scale_factor)
        H = dscale/scale_factor[:-1]
        dic = {'time': time[:-1],
               'a': scale_factor[:-1],
               'dadt': dscale,
               'H': H,
               'part_h': part_h,
               'evt_h': evt_h
               }
        self.cosmo = None
        self.cosmo_tab = pd.DataFrame(dic)
        self.cosmo_tab.to_csv(output_file)
        mask = self.cosmo_tab['time'] < self.t_amax
        self.t_today = interp_nb(1,
                                 self.cosmo_tab['a'][mask].to_numpy(),
                                 self.cosmo_tab['time'][mask].to_numpy())

    @staticmethod
    def compute_event_h(time, scale_factor):
        evt_h = compute_evt_h(time, scale_factor)
        return evt_h

    def plot(self, d1, d2, xlim=None, ylim=None):
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

        self.parts_coords = [r0, ang0, r0 * np.cos(ang0), r0 * np.sin(ang0)]
        self.photons_coords = [r0, ang0, r0 * np.cos(ang0), r0 * np.sin(ang0), np.ones(len(r0))]

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
                                     self.dt)

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
        self.photons.set_color(self.cmap(self.a_em/self.a_t))

        self.hubble_h_plot.set_data(1/self.H * self.circle[0] * self.norm_factor,
                                    1/self.H * self.circle[1] * self.norm_factor)
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
