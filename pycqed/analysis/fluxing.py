from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel
from pycqed.analysis.tools.plotting import flex_colormesh_plot_vs_xy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pycqed.analysis import composite_analysis as ca
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis.measurement_analysis import MeasurementAnalysis
import warnings
import matplotlib.pyplot as plt
#############################
from pycqed.analysis import fitting_models as fit_mods
import lmfit
import numpy as np


def omega(flux, f_max, EC, asym):
    return fit_mods.Qubit_dac_to_freq(flux, f_max, E_c=EC,
                                      dac_sweet_spot=0, dac_flux_coefficient=1, asymmetry=0)


f_flux = lambda flux: omega(flux=flux,
                            f_max=6.10e9,
                            EC=0.28e9,
                            asym=0.)-0.28e9


def ChevFourierFunc(delta, alpha, beta, g, branch):
    assert(len(delta) == len(branch))
    freqs = alpha*np.sqrt(4*g*g+beta*beta*delta*delta)
    return np.where(branch, freqs, -freqs)


def ChevFourierFunc2(delta, alpha, beta, f_res, g, branch):
    assert(len(delta) == len(branch))
    freqs = alpha*np.sqrt(4*g*g+4.*np.pi*np.pi*(f_flux(beta*delta)-f_res)**2)
    return np.where(branch, freqs, -freqs)


def reshape_axis_2d(axis_array):
    x = axis_array[0, :]
    y = axis_array[1, :]
    dimx = np.sum(np.where(x == x[0], 1, 0))
    dimy = len(x) // dimx
    if dimy*dimx < len(x):
        warnings.warn('Data was cut-off. Probably due to an interrupted scan')
        dimy_c = dimy + 1
    else:
        dimy_c = dimy
    return x[:dimy_c], (y[::dimy_c])


def reshape_data(sweep_points, data):
    x, y = reshape_axis_2d(sweep_points)
    dimx = len(x)
    dimy = len(y)
    dim = dimx*dimy
    if dim > len(data):
        dimy = dimy - 1
    return x, y[:dimy], (data[:dimx*dimy].reshape((dimy, dimx))).transpose()


class Chevron_2D(MeasurementAnalysis):

    def run_default_analysis(self, save_fig=True, **kw):
        self.get_naming_and_values_2D()

        # Getting the data in the right shape
        x = self.sweep_points
        y = self.sweep_points_2D
        z = self.measured_values[0].transpose()

        self.plot_times = x
        self.plot_step = self.plot_times[1]-self.plot_times[0]

        self.plot_x = y
        self.x_step = self.plot_x[1]-self.plot_x[0]
        result = z

        self.x = x
        self.y = y
        self.z = z

        self.make_figures(save_fig=save_fig, **kw)

    def make_figures(self, save_fig=True, **kw):

        f, axs = plt.subplots(1, 2, figsize=(10, 4))

        cmin, cmax = 0, 1
        fig_clim = [cmin, cmax]

        out = a_tools.color_plot(self.x, self.y, self.z,
                                 fig=f, ax=axs[0], add_colorbar=True,
                                 zlabel='Qubit excitation probability')
        set_xlabel(axs[0], self.parameter_names[0], self.parameter_units[0])
        set_ylabel(axs[0], self.parameter_names[1], self.parameter_units[1])
        figname = '{}: Chevron scan'.format(self.timestamp)
        axs[0].set_title(figname)

        ax = axs[1]
        plot_fft = np.fft.fft(result[:-4, :], axis=0)
        plot_fft_f = np.fft.fftfreq(len(self.plot_x[:-4]), self.x_step)
        fft_step = plot_fft_f[1]-plot_fft_f[0]
        sort_vec = np.argsort(plot_fft_f)
        plot_fft_abs = np.abs(plot_fft[sort_vec, :])

        y = plot_fft_f[sort_vec]/2*np.pi
        mask_higher = np.where(y > 2.*(y[1]-y[0]), True, False)
        mask_lower = np.where(y < 2.*(y[0]-y[1]), True, False)

        peaks_higher = np.zeros(len(self.plot_times))
        peaks_lower = np.zeros(len(self.plot_times))
        for i, p in enumerate(self.plot_times):
            u = y[mask_higher]
            peaks_higher[i] = u[np.argmax(plot_fft_abs[mask_higher, i])]
            u = y[mask_lower]
            peaks_lower[i] = u[np.argmax(plot_fft_abs[mask_lower, i])]

        cmin, cmax = None, None  # 0, 10.
        fig_clim = [cmin, cmax]
        out = flex_colormesh_plot_vs_xy(ax=ax, clim=fig_clim, cmap='viridis',
                                        xvals=self.plot_times,
                                        yvals=y,
                                        zvals=plot_fft_abs)
        ax.plot(self.plot_times, peaks_lower,
                'o', fillstyle='none', c='orange')
        ax.plot(self.plot_times, peaks_higher,
                'o', fillstyle='none', c='orange')
        set_xlabel(ax, self.parameter_names[0], self.parameter_units[0])
        set_ylabel(ax, 'Frequency', 'Hz')
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes('right', size='10%', pad='5%')
        cbar = plt.colorbar(out['cmap'], cax=cax)
        cbar.set_label('Fourier transform')

        # Do fitting and plot fit results
        fit_res, fit_res.best_values = self.fit_fourier_transform(
            self.plot_times, peaks_lower, peaks_higher)
        # eval_fit_lower = lambda d: ChevFourierFunc2(
        #     delta=d,
        #     **fit_res.best_values, branch=np.zeros(d.shape, dtype=np.bool))
        # eval_fit_higher = lambda d: ChevFourierFunc2(
        #     delta=d,
        #     **fit_res.best_values, branch=np.ones(d.shape, dtype=np.bool))
        eval_fit_lower = lambda d: ChevFourierFunc(
            delta=d,
            **fit_res.best_values, branch=np.zeros(d.shape, dtype=np.bool))
        eval_fit_higher = lambda d: ChevFourierFunc(
            delta=d,
            **fit_res.best_values, branch=np.ones(d.shape, dtype=np.bool))



        ax.plot(self.plot_times, eval_fit_lower(self.plot_times), '-',
                c='orange', label='fit')
        ax.plot(self.plot_times, eval_fit_higher(self.plot_times), '-',
                c='orange')
        ax.set_xlim(self.plot_times.min(), self.plot_times.max())

        coupling_label = '$J_2$'
        g_legend = r'{} = {:.2f}$\pm${:.2f} MHz'.format(
            coupling_label,
            fit_res.params['g']/(2*np.pi), fit_res.params['g'].stderr/(2*np.pi))
        ax.text(.5, .8, g_legend, transform=ax.transAxes, color='white')

        f.tight_layout()
        if save_fig:
            self.save_fig(f, figname='test', close_fig=False)

    def fit_fourier_transform(self, times, peaks_lower, peaks_higher):
        fit_mask_lower = np.array([True]*len(peaks_lower))
        fit_mask_higher = np.array([True]*len(peaks_higher))
        # Hardcoded, not cool
        # fit_mask_lower[:25] = False
        # fit_mask_higher[:25] = False
        print(times)

        my_fit_points = np.concatenate(
            (times[fit_mask_lower], times[fit_mask_higher]))
        my_fit_data = np.concatenate(
            (peaks_lower[fit_mask_lower], peaks_higher[fit_mask_higher]))
        mask_branch = np.concatenate((np.zeros(len(peaks_lower[fit_mask_lower]), dtype=np.bool),
                                      np.ones(len(peaks_higher[fit_mask_higher]), dtype=np.bool)))

        fit_func = lambda delta, alpha, beta, f_res, g: ChevFourierFunc2(
            delta, alpha, beta, f_res, g, mask_branch)
        fit_func = lambda delta, alpha, beta, g: ChevFourierFunc(
            delta, alpha, beta, g, mask_branch)
        ChevFourierModel = lmfit.Model(fit_func)

        ChevFourierModel.set_param_hint(
            'alpha', value=1./1000, min=0.)#, max=10.e6, vary=True)
        ChevFourierModel.set_param_hint(
            'beta', value=0.14/1000)#, min=-10., max=10., vary=True)
        # ChevFourierModel.set_param_hint(
        #     'f_res', value=4.68e9, min=0., max=50.0e9, vary=False)
        ChevFourierModel.set_param_hint(
            'g', value=np.pi*2e6, min=0, max=500e6)
            # 0.0239*2.0e9, min=0, max=2000, vary=True)

        my_fit_params = ChevFourierModel.make_params()

        self.fit_res = ChevFourierModel.fit(
            data=my_fit_data, delta=my_fit_points, params=my_fit_params)
        print(self.fit_res.fit_report())
        self.fit_res.plot_fit()
        return self.fit_res, self.fit_res.best_values


a = Chevron_2D()

x = a.y
y = a.x
z = a.z
print('hello')

# scan_start = '20170421_102001'
# scan_stop = '20170421_222001'

# pdict = {'I': 'amp',
#          'sweep_points': 'sweep_points'}
# opt_dict = {'scan_label': 'chevron'}
# nparams = ['I', 'sweep_points']
# spec_scans = ca.quick_analysis(t_start=scan_start, t_stop=scan_stop, options_dict=opt_dict,
#                                params_dict_TD=pdict, numeric_params=nparams)
# x, y, z = reshape_data(
#     spec_scans.TD_dict['sweep_points'][0], spec_scans.TD_dict['I'][0])

# y, x, z = reshape_data(
#     spec_scans.TD_dict['sweep_points'][0], spec_scans.TD_dict['I'][0])
# z = z.T


# Basic figures
# fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# ax = axs[0]
# plot_times = y
# plot_step = plot_times[1]-plot_times[0]

# plot_x = x*1e9
# x_step = plot_x[1]-plot_x[0]

result = z

# cmin, cmax = None, None  # 0, 1.
# fig_clim = [cmin, cmax]
# out = flex_colormesh_plot_vs_xy(ax=ax, clim=fig_clim, cmap='viridis',
#                                 xvals=plot_times,
#                                 yvals=plot_x,
#                                 zvals=result)
# ax.set_xlabel(r'AWG Amp (Vpp)')
# ax.set_ylabel(r'Time (ns)')
# # ax.set_xlim(xmin, xmax)
# ax.set_ylim(plot_x.min()-x_step/2., plot_x.max()+x_step/2.)
# ax.set_xlim(plot_times.min()-plot_step/2., plot_times.max()+plot_step/2.)
# ax_divider = make_axes_locatable(ax)
# cax = ax_divider.append_axes('right', size='10%', pad='5%')
# cbar = plt.colorbar(out['cmap'], cax=cax)
# cbar.set_label('Qubit excitation probability')

# ax.title.set_fontsize(14)
# fig.savefig(filename=save_name+'.png',format='png')

# fig.tight_layout()


# ax = axs[1]
# plot_fft = np.fft.fft(result[:-4, :], axis=0)
# plot_fft_f = np.fft.fftfreq(len(plot_x[:-4]), x_step)
# fft_step = plot_fft_f[1]-plot_fft_f[0]
# sort_vec = np.argsort(plot_fft_f)
# plot_fft_abs = np.abs(plot_fft[sort_vec, :])

# y = plot_fft_f[sort_vec]/2*np.pi
# mask_higher = np.where(y > 2.*(y[1]-y[0]), True, False)
# mask_lower = np.where(y < 2.*(y[0]-y[1]), True, False)

# peaks_higher = np.zeros(len(plot_times))
# peaks_lower = np.zeros(len(plot_times))
# for i, p in enumerate(plot_times):
#     u = y[mask_higher]
#     peaks_higher[i] = u[np.argmax(plot_fft_abs[mask_higher, i])]
#     u = y[mask_lower]
#     peaks_lower[i] = u[np.argmax(plot_fft_abs[mask_lower, i])]

# cmin, cmax = None, None  # 0, 10.
# fig_clim = [cmin, cmax]
# out = flex_colormesh_plot_vs_xy(ax=ax, clim=fig_clim, cmap='viridis',
#                                 xvals=plot_times,
#                                 yvals=y,
#                                 zvals=plot_fft_abs)
# ax.plot(plot_times, peaks_lower, 'o', fillstyle='none', c='orange')
# ax.plot(plot_times, peaks_higher, 'o', fillstyle='none', c='orange')
# ax.set_xlabel(r'Amplitude (Vpp)')
# ax.set_ylabel(r'Time (ns)')
# # ax.set_xlim(xmin, xmax)
# ax.set_ylim(plot_fft_f.min()-fft_step/2., plot_fft_f.max()+fft_step/2.)
# ax.set_xlim(plot_times.min()-plot_step/2., plot_times.max()+plot_step/2.)
# # ax.set_xlim(0,50)
# ax_divider = make_axes_locatable(ax)
# cax = ax_divider.append_axes('right', size='10%', pad='5%')
# cbar = plt.colorbar(out['cmap'], cax=cax)
# # cbar.set_ticks(np.arange(fig_clim[0],1.01*fig_clim[1],(fig_clim[1]-fig_clim[0])/5.))
# # cbar.set_ticklabels([str(fig_clim[0]),'','','','',str(fig_clim[1])])
# cbar.set_label('Fourier transform')

# fig.tight_layout()
# fit_mask_lower = np.array([True]*len(peaks_lower))
# fit_mask_higher = np.array([True]*len(peaks_higher))
# fit_mask_lower[:25] = False
# fit_mask_higher[:25] = False

# my_fit_points = np.concatenate(
#     (plot_times[fit_mask_lower], plot_times[fit_mask_higher]))
# my_fit_data = np.concatenate(
#     (peaks_lower[fit_mask_lower], peaks_higher[fit_mask_higher]))
# mask_branch = np.concatenate((np.zeros(len(peaks_lower[fit_mask_lower]), dtype=np.bool),
# np.ones(len(peaks_higher[fit_mask_higher]), dtype=np.bool)))

# fit_func = lambda delta, alpha, beta, f_res, g: ChevFourierFunc2(
#     delta, alpha, beta, f_res, g, mask_branch)
# ChevFourierModel = lmfit.Model(fit_func)

# ChevFourierModel.set_param_hint('alpha', value=1., min=0., max=10., vary=True)
# ChevFourierModel.set_param_hint(
#     'beta', value=0.14, min=-10., max=10., vary=True)
# ChevFourierModel.set_param_hint(
#     'f_res', value=4.68, min=0., max=50., vary=False)
# ChevFourierModel.set_param_hint(
#     'g', value=np.pi*0.0239*2., min=0, max=2000, vary=True)

# my_fit_params = ChevFourierModel.make_params()

# fit_res = ChevFourierModel.fit(
#     data=my_fit_data, delta=my_fit_points, params=my_fit_params)
# eval_fit_lower = lambda d: ChevFourierFunc2(delta=d, **fit_res.best_values, branch=np.zeros(d.shape, dtype=np.bool))
# eval_fit_higher = lambda d: ChevFourierFunc2(delta=d, **fit_res.best_values, branch=np.ones(d.shape, dtype=np.bool))
# fit_res, fit_res.best_values
# print(fit_res.fit_report())

# # ax.plot(plot_times[fit_mask_lower],peaks_lower[fit_mask_lower],'ro')
# # ax.plot(plot_times[fit_mask_higher],peaks_higher[fit_mask_higher],'ro')
# ax.plot(plot_times, eval_fit_lower(plot_times), '-', c='orange', label='fit')
# ax.plot(plot_times, eval_fit_higher(plot_times), '-', c='orange')
# ax.set_xlim(plot_times.min(), plot_times.max())

# coupling_label = '$J_2$'
# g_legend = r'{} = {:.2f}$\pm${:.2f} MHz'.format(
#     coupling_label,
#     fit_res.params['g']/(2*np.pi)*1e3, fit_res.params['g'].stderr/(2*np.pi)*1e3)
# ax.text(.6, .8, g_legend, transform=ax.transAxes, color='white')


# class Chevron_2D(MeasurementAnalysis):

#     def run_default_analysis(self, save_fig=True, **kw):
#         self.get_naming_and_values_2D()

#         x = self.sweep_points
#         y = self.sweep_points_2D
#         z = self.measured_values[0].transpose()

#         plot_times = y
#         plot_step = plot_times[1]-plot_times[0]
#         plot_x = x
#         x_step = plot_x[1]-plot_x[0]
#         result = z

#         fig = plt.figure(figsize=(8, 6))
#         ax = fig.add_subplot(111)
#         cmin, cmax = 0, 1
#         fig_clim = [cmin, cmax]

#         out = a_tools.color_plot(x, y, z, fig=fig, ax=ax, add_colorbar=True,
#                                  zlabel='Qubit excitation probability')
#         set_xlabel(ax, self.parameter_names[0], self.parameter_units[0])
#         set_ylabel(ax, self.parameter_names[1], self.parameter_units[1])
#         figname = '{}: Chevron scan'.format(self.timestamp)
#         ax.set_title(figname)

#         fig.tight_layout()
#         if save_fig:
#             self.save_fig(fig, figname='test')

#     def reshape_axis_2d(self, axis_array):
#         x = axis_array[0, :]
#         y = axis_array[1, :]
#         # print(y)
#         dimx = np.sum(np.where(x == x[0], 1, 0))
#         dimy = len(x) // dimx
#         # print(dimx,dimy)
#         if dimy*dimx < len(x):
#             logging.warning.warn(
#                 'Data was cut-off. Probably due to an interrupted scan')
#             dimy_c = dimy + 1
#         else:
#             dimy_c = dimy
#         return x[:dimy_c], (y[::dimy_c])

#     def reshape_data(self, sweep_points, data):
#         x, y = self.reshape_axis_2d(sweep_points)
#         # print(x,y)
#         dimx = len(x)
#         dimy = len(y)
#         dim = dimx*dimy
#         if dim > len(data):
#             dimy = dimy - 1
# return x, y[:dimy], (data[:dimx*dimy].reshape((dimy, dimx))).transpose()

#     def save_fig(self, fig, figname=None, xlabel='x', ylabel='y',
#                  fig_tight=True, **kw):
#         plot_formats = kw.pop('plot_formats', ['png'])
#         fail_counter = False
#         close_fig = kw.pop('close_fig', True)
#         if type(plot_formats) == str:
#             plot_formats = [plot_formats]
#         for plot_format in plot_formats:
#             if figname is None:
#                 figname = (self.timestamp+'_Chevron_2D_'+'.'+plot_format)
#             else:
#                 figname = (figname+'.' + plot_format)
#             self.savename = os.path.abspath(os.path.join(
#                 self.folder, figname))
#             if fig_tight:
#                 try:
#                     fig.tight_layout()
#                 except ValueError:
#                     print('WARNING: Could not set tight layout')
#             try:
#                 fig.savefig(
#                     self.savename, dpi=300,
#                     # value of 300 is arbitrary but higher than default
#                     format=plot_format)
#             except:
#                 fail_counter = True
#         if fail_counter:
#             logging.warning('Figure "%s" has not been saved.' % self.savename)
#         if close_fig:
#             plt.close(fig)
#         return
