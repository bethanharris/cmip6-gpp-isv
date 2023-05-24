#!/usr/bin/env python
# Copyright: This document has been placed in the public domain.

"""
Taylor diagram (Taylor, 2001) implementation.
Note: If you have found these software useful for your research, I would
appreciate an acknowledgment.
"""

__version__ = "Time-stamp: <2018-12-06 11:43:41 ycopin>"
__author__ = "Yannick Copin <yannick.copin@laposte.net>"

import numpy as NP
import matplotlib.pyplot as PLT


class TaylorDiagram(object):
    """
    Taylor diagram.
    Plot model standard deviation and correlation to reference (data)
    sample in a single-quadrant polar plot, with r=stddev and
    theta=arccos(correlation).
    """

    def __init__(self, refstd,
                 fig=None, rect=111, label='_', srange=(0, 1.5), extend=False,
                 normalised_stdev=False, sqrt_stdev=False, rotate_stdev_labels=False):
        """
        Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using `mpl_toolkits.axisartist.floating_axes`.
        Parameters:
        * refstd: reference standard deviation to be compared to
        * fig: input Figure or None
        * rect: subplot definition
        * label: reference label
        * srange: stddev axis extension, in units of *refstd*
        * extend: extend diagram to negative correlations
        """

        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        self.refstd = refstd            # Reference standard deviation

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = NP.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        if extend:
            # Diagram extended to negative correlations
            self.tmax = NP.pi
            rlocs = NP.concatenate((-rlocs[:0:-1], rlocs))
        else:
            # Diagram limited to positive correlations
            self.tmax = NP.pi/2
        tlocs = NP.arccos(rlocs)        # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)    # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        if sqrt_stdev:
            slocs = NP.array([0, 0.5, 1, NP.sqrt(2), NP.sqrt(3), NP.sqrt(4)])
            actual_values = NP.copy(slocs)
            actual_values[slocs>1] = slocs[slocs>1]**2
            int_diff = NP.abs(actual_values - NP.round(actual_values))
            actual_values[int_diff<1e-6] = NP.round(actual_values[int_diff<1e-6])
            gl2 = GF.FixedLocator(slocs)
            tf2 = GF.DictFormatter(dict(zip(slocs, map(str, actual_values))))
        else:
            gl2 = None
            tf2 = None


        # Standard deviation axis extent (in units of reference stddev)
        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd

        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1,
            grid_locator2=gl2, tick_formatter2=tf2)

        if fig is None:
            fig = PLT.figure()

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")   # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].major_ticklabels.set_size(12)
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")
        ax.axis["top"].label.set_size(14)

        ax.axis["left"].set_axis_direction("bottom")  # "X axis"
        if normalised_stdev:
            ax.axis["left"].label.set_text(r"$\sigma_{\mathrm{model}}/\sigma_{\mathrm{obs}}$")
        else:
            ax.axis["left"].label.set_text("Standard deviation")
        ax.axis["left"].label.set_size(16)
        ax.axis["left"].major_ticks.set_tick_out(True)
        ax.axis["left"].major_ticks.set_ticksize(5)
        ax.axis["left"].major_ticklabels.set_pad(5)
        if rotate_stdev_labels:
            ax.axis["left"].major_ticklabels.set_rotation(20)
            ax.axis["left"].label.set_pad(7)
        ax.axis["left"].major_ticklabels.set_size(12)
        ax.axis["right"].set_axis_direction("top")    # "Y-axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if extend else "left")
        ax.axis["right"].major_ticks.set_tick_out(True)
        ax.axis["right"].major_ticklabels.set_size(12)

        if extend:
            ax.axis["right"].major_ticks.set_ticksize(5)
            ax.axis["right"].major_ticklabels.set_pad(5)
            ax.axis["right"].major_ticklabels.set_size(12)
    
        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)          # Unused

        self._ax = ax                   # Graphical axes
        self.ax = ax.get_aux_axes(tr)   # Polar coordinates

        # Add reference point and stddev contour
        # l, = self.ax.plot([0], self.refstd, 'k*',
        #                   ls='', ms=10, label=label, clip_on=False)
        # t = NP.linspace(0, self.tmax)
        # r = NP.zeros_like(t) + self.refstd
        # self.ax.plot(t, r, 'k--', label='_')

        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = []

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """
        Add sample (*stddev*, *corrcoeff*) to the Taylor
        diagram. *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """

        l, = self.ax.plot(NP.arccos(corrcoef), stddev, zorder=3,
                          *args, **kwargs)  # (theta, radius)
        self.samplePoints.append(l)

        return l

    def add_grid(self, *args, **kwargs):
        """Add a grid."""

        self._ax.grid(*args, **kwargs)

    def add_contours(self, levels=5, sqrt_stdev=False, **kwargs):
        """
        Add constant centered RMS difference contours, defined by *levels*.
        """

        rs, ts = NP.meshgrid(NP.linspace(self.smin, self.smax),
                             NP.linspace(0, self.tmax))
        if sqrt_stdev:
            actual_rs = NP.copy(rs)
            actual_rs[rs>1.] = (rs[rs>1.])**2
            rms = NP.sqrt(self.refstd**2 + actual_rs**2 - 2*self.refstd*actual_rs*NP.cos(ts))
        # Compute centered RMS difference
        else:
            rms = NP.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*NP.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

        return contours

    def add_refstd(self, stddev, marker, label):
        l, = self.ax.plot([0], stddev, marker=marker, mfc='k', mec='0.5',
                          ls='', ms=10, label=label, clip_on=False, zorder=3)
        self.samplePoints.append(l)
        t = NP.linspace(0, self.tmax)
        r = NP.zeros_like(t) + stddev
        self.ax.plot(t, r, 'k--', label='_')
