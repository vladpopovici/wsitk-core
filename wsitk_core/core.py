#   -*- coding: utf-8 -*-
#
#  --------------------------------------------------------------------
#  Copyright (c) 2022-2023 Vlad Popovici <popovici@bioxlab.org>
#
#  Licensed under the MIT License. See LICENSE file in root folder.
#  --------------------------------------------------------------------

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = "0.1"

"""
Core WSI-related functions and classes. The class WSI provides a set
of useful functions for reading images from a WSI file (via openslide)
and performing relevant conversions resolution <-> pyramid level <-> mpp.
The main dependencies are <openslide> and <shapely> packages.
"""

__all__ = ['WSI', 'ImageShape', 'Magnification']

import pathlib
import numpy as np
from math import log
from typing import Optional, NewType, Tuple, Any
import openslide as osl
import shapely.geometry as shg
import shapely.affinity as sha

ImageShape = NewType("ImageShape", dict[str, int])

#####
class Magnification:
    def __init__(self,
                 magnif: float,
                 mpp: float,
                 level: int = 0,
                 n_levels: int = 10,
                 magnif_step: float = 2.0):
        """Magnification handling/conversion.

        Args:
            magnif: base objective magnification (e.g. 10.0)
            mpp: resolution in microns per pixel at the given magnification
                (e.g. 0.245).
            level: level in the image pyramid corresponding to the
                magnification/mpp. Defaults to 0 - highest magnification.
            n_levels: number of levels in the image pyramid that are relevant/
                feasible
            magnif_step: scaling factor between levels in the image pyramid.
        """
        if level < 0 or level >= n_levels:
            raise RuntimeError("Specified level outside [0, (n_levels-1)]")
        self._base_magnif = magnif
        self._base_mpp = mpp
        self._base_level = level
        self._magnif_step = magnif_step
        self._n_levels = n_levels
        # initialize look-up tables:
        self.__magnif = magnif * self._magnif_step ** (level - np.arange(n_levels))
        self.__mpp = mpp * self._magnif_step ** (np.arange(n_levels) - level)

        return

    @property
    def magnif_step(self) -> float:
        return self._magnif_step

    def get_magnif_for_mpp(self, mpp: float) -> float:
        """
        Returns the objective magnification for a given resolution.
        Args:
            mpp: target resolution (microns per pixel)

        Returns:
            float: magnification corresponding to mpp. If <mpp> is outside the
                normal interval then return the corresponding end of the
                magnification values if still close enough (relative error <=0.1)
                or raise an Error
        """
        if mpp < self.__mpp[0]:
            # mpp outside normal interval, try to see if it's too far:
            if (self.__mpp[0] - mpp) / self.__mpp[0] > 0.1:
                raise RuntimeError('mpp outside supported interval') from None
            else:
                return float(self.__magnif[0])
        if mpp > self.__mpp[self._n_levels - 1]:
            # mpp outside normal interval, try to see if it's too far:
            if (mpp - self.__mpp[self._n_levels - 1]) / self.__mpp[self._n_levels - 1] > 0.1:
                raise RuntimeError('mpp outside supported interval') from None
            else:
                return float(self.__magnif[self._n_levels - 1])
        k = np.argmin(np.abs(mpp - self.__mpp))

        return float(self.__magnif[k])

    def get_mpp_for_magnif(self, magnif: float) -> float:
        """
        Return the resolution (microns per pixel - mpp) for a given objective
            magnification.
        Args:
            magnif: target magnification

        Returns:
            float: resolution (microns per pixel) corresponding to magnification
        """
        if magnif > self.__magnif[0] or magnif < self.__magnif[self._n_levels - 1]:
            raise RuntimeError('magnif outside supported interval') from None
        k = np.argmin(np.abs(magnif - self.__magnif))

        return float(self.__mpp[k])

    def get_level_for_magnif(self, magnif: float) -> int:
        """
        Return the level for a given objective magnification. Negative values
        correspond to magnification levels higher than the indicated base level.

        Args:
            magnif: target magnification

        Returns:
            int: resolution (mpp) corresponding to magnification
        """
        if magnif > self.__magnif[0] or magnif < self.__magnif[self._n_levels - 1]:
            raise RuntimeError('magnif outside supported interval') from None

        k = np.argmin(np.abs(magnif - self.__magnif))

        return k

    def get_level_for_mpp(self, mpp: float) -> int:
        """
        Return the level for a given resolution. Negative values
        correspond to resolution levels higher than the indicated base level.

        Args:
            mpp: target resolution

        Returns:
            int: resolution (mpp) corresponding to magnification
        """
        if mpp < self.__mpp[0]:
            # mpp outside normal interval, try to see if it's too far:
            if (self.__mpp[0] - mpp) / self.__mpp[0] > 0.1:
                raise RuntimeError('mpp outside supported interval') from None
            else:
                return 0
        if mpp > self.__mpp[self._n_levels - 1]:
            # mpp outside normal interval, try to see if it's too far:
            if (mpp - self.__mpp[self._n_levels - 1]) / self.__mpp[self._n_levels - 1] > 0.1:
                raise RuntimeError('mpp outside supported interval') from None
            else:
                return self._n_levels - 1

        k = np.argmin(np.abs(mpp - self.__mpp))

        return k

    def get_mpp_for_level(self, level: int) -> float:
        """
        Return the resolution (mpp) for a given level.

        Args:
            level: target level

        Returns:
            float: resolution (mpp)
        """
        if level < 0 or level >= self._n_levels:
            raise RuntimeError('level outside supported interval.') from None

        return float(self.__mpp[level])

    def get_magnification_step(self) -> float:
        """
        Return the magnification step between two consecutive levels.

        Returns:
            float: magnification step
        """
        return self._magnif_step

#####
class WSI(object):
    """An extended version of OpenSlide, with more handy methods
    for dealing with microscopy slide images.

    Args:
        path (str): full path to the image file

    Attributes:
        _path (str): full path to WSI file
        info (dict): slide image metadata
    """

    def __init__(self, path: str | pathlib.Path):
        self._path = pathlib.Path(path) if isinstance(path, str) else path
        self._slide = slide_src = osl.OpenSlide(self.path)
        self._original_meta = slide_meta = slide_src.properties
        self.info = {
            'objective_power': float(slide_meta[osl.PROPERTY_NAME_OBJECTIVE_POWER]),
            'width':  slide_src.dimensions[0],
            'height': slide_src.dimensions[1],
            'mpp_x': float(slide_meta[osl.PROPERTY_NAME_MPP_X]), # microns/pixel
            'mpp_y': float(slide_meta[osl.PROPERTY_NAME_MPP_Y]),
            'n_levels': slide_src.level_count,    # no. of levels in pyramid
            'magnification_step': slide_src.level_downsamples[1] / slide_src.level_downsamples[0],
            'roi': None,
            'background': 0xFF
        }

        # optional properties:
        if osl.PROPERTY_NAME_BOUNDS_X in slide_meta:
            self.info['roi'] = {
                'x0': int(slide_meta[osl.PROPERTY_NAME_BOUNDS_X]),
                'y0': int(slide_meta[osl.PROPERTY_NAME_BOUNDS_Y]),
                'width': int(slide_meta[osl.PROPERTY_NAME_BOUNDS_WIDTH]),
                'height': int(slide_meta[osl.PROPERTY_NAME_BOUNDS_HEIGHT]),
            }
        if osl.PROPERTY_NAME_BACKGROUND_COLOR in slide_meta:
            self.info['background'] = 0xFF if slide_meta[osl.PROPERTY_NAME_BACKGROUND_COLOR] == 'FFFFFF' else 0

        # _pyramid_levels: 2 x n_levels: [max_x, max_y] x [0,..., n_levels-1]
        self._pyramid_levels = np.zeros((2, self.info['n_levels']), dtype=int)

        # _pyramid_mpp: 2 x n_levels: [mpp_x, mpp_y] x [0,..., n_levels-1]
        self._pyramid_mpp = np.zeros((2, self.info['n_levels']))
        self._downsample_factors = np.zeros((self.info['n_levels'],), dtype=int)

        self.magnif_converter =  Magnification(
            self.info['objective_power'],
            mpp=0.5*(self.info['mpp_x'] + self.info['mpp_y']),
            level=0,
            magnif_step=float(self.info['magnification_step'])
        )

        for lv in range(self.info['n_levels']):
            s = 2.0**lv
            self._pyramid_levels[0, lv] = slide_src.level_dimensions[lv][0]
            self._pyramid_levels[1, lv] = slide_src.level_dimensions[lv][1]
            self._pyramid_mpp[0, lv] = s * self.info['mpp_x']
            self._pyramid_mpp[1, lv] = s * self.info['mpp_y']
            self._downsample_factors[lv] = self.magnif_converter.magnif_step ** lv

        return

    @property
    def level_count(self) -> int:
        """Return the number of levels in the multi-resolution pyramid."""
        return self.info['n_levels']


    def downsample_factor(self, level:int) -> int:
        """Return the down-sampling factor (relative to level 0) for a given level."""
        if level < 0 or level >= self.level_count:
            return -1
        ms_x = self._pyramid_levels[0, level] / self.info['mpp_x']
        ms_y = self._pyramid_levels[1, level] / self.info['mpp_y']

        ms = round(0.5*(ms_x + ms_y))

        return int(ms)

    @property
    def get_native_magnification(self) -> float:
        """Return the original magnification for the scan."""
        return self.info['objective_power']

    @property
    def get_native_resolution(self) -> float:
        """Return the scan resolution (microns per pixel)."""
        return 0.5 * (self.info['mpp_x'] + self.info['mpp_y'])

    def get_level_for_magnification(self, mag: float, eps=1e-6) -> int:
        """Returns the level in the image pyramid that corresponds the given magnification.

        Args:
            mag (float): magnification
            eps (float): accepted error when approximating the level

        Returns:
            level (int) or -1 if no suitable level was found
        """
        if mag > self.info['objective_power'] or mag < 2.0**(1-self.level_count) * self.info['objective_power']:
            return -1

        lx = log(self.info['objective_power'] / mag, self.info['magnification_step'])
        k = np.where(np.isclose(lx, range(0, self.level_count), atol=eps))[0]
        if len(k) > 0:
            return int(k[0])   # first index matching
        else:
            return -1   # no match close enough

    def get_level_for_mpp(self, mpp: float):
        """Return the level in the image pyramid that corresponds to a given resolution."""
        return self.magnif_converter.get_level_for_mpp(mpp)

    def get_mpp_for_level(self, level: int):
        """Return resolution (mpp) for a given level in pyramid."""
        return self.magnif_converter.get_mpp_for_level(level)

    def get_magnification_for_level(self, level: int) -> float:
        """Returns the magnification (objective power) for a given level.

        Args:
            level (int): level in the pyramidal image

        Returns:
            magnification (float)
            If the level is out of bounds, returns -1.0
        """
        if level < 0 or level >= self.level_count:
            return -1.0
        if level == 0:
            return self.info['objective_power']

        #return 2.0**(-level) * self.info['objective_power']
        return self.info['magnification_step'] ** (-level) * self.info['objective_power']

    def get_extent_at_level(self, level: int) -> Optional[ImageShape]:
        """Returns width and height of the image at a desired level.

        Args:
            level (int): level in the pyramidal image

        Returns:
            (width, height) of the level
        """
        if level < 0 or level >= self.level_count:
            return None
        return ImageShape({
            'width': self._pyramid_levels[0, level],
            'height': self._pyramid_levels[1, level]
        })

    @property
    def pyramid_levels(self):
        return self._pyramid_levels

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @property
    def widths(self) -> np.array:
        # All widths for the pyramid levels
        return self._pyramid_levels[0, :]

    @property
    def heights(self) -> np.array:
        # All heights for the pyramid levels
        return self._pyramid_levels[1, :]

    def extent(self, level: int = 0) -> tuple[Any, ...]:
        # width, height for a given level
        return tuple(self._pyramid_levels[:, level])

    def level_shape(self, level: int = 0) -> ImageShape:
        return ImageShape(
            {
                'width': int(self._pyramid_levels[0, level]),
                'height': int(self._pyramid_levels[1, level])
            }
        )

    def between_level_scaling_factor(self, from_level: int, to_level: int) -> float:
        """Return the scaling factor for converting coordinates (magnification)
        between two levels in the MRI.

        Args:
            from_level (int): original level
            to_level (int): destination level

        Returns:
            float
        """
        f = self._downsample_factors[from_level] / self._downsample_factors[to_level]

        return float(f)

    def convert_px(self, point, from_level, to_level) -> Tuple[int, int]:
        """Convert pixel coordinates of a point from <from_level> to
        <to_level>

        Args:
            point (tuple): (x,y) coordinates in <from_level> plane
            from_level (int): original image level
            to_level (int): destination level

        Returns:
            x, y (float): new coordinates - no rounding is applied
        """
        if from_level == to_level:
            return point  # no conversion is necessary
        x, y = point
        f = self.between_level_scaling_factor(from_level, to_level)
        x *= f
        y *= f

        return int(x), int(y)

    def get_region_px(self, x0: int, y0: int,
                      width: int, height: int,
                      level: int = 0, as_type=np.uint8) -> np.ndarray:
        """Read a region from the image source. The region is specified in
            pixel coordinates.

            Args:
                x0, y0 (int): top left corner of the region (in pixels, at the specified
                level)
                width, height (int): width and height (in pixels) of the region.
                level (int): the level in the image pyramid to read from
                as_type: type of the pixels (default numpy.uint8)

            Returns:
                a numpy.ndarray [height x width x channels]
        """
        if level < 0 or level >= self.level_count:
            raise RuntimeError("requested level does not exist")

        # check bounds:
        if x0 >= self.widths[level] or y0 >= self.heights[level] or \
                x0 + width > self.widths[level] or \
                y0 + height > self.heights[level]:
            raise RuntimeError("region out of layer's extent")

        x0_0, y0_0 = self.convert_px((x0, y0), level, 0)
        img = self._slide.read_region((x0_0, y0_0), level, (width, height))
        img = np.array(img)

        if img.shape[2] == 4:  # has alpha channel, usually used for masking
            # fill with background
            mask = img[..., -1].squeeze()
            img[mask == 0, 0:4] = self.info['background']
            img = img[..., :-1]

        return img.astype(as_type)

    def get_plane(self, level: int = 0, as_type=np.uint8) -> np.ndarray:
        """Read a whole plane from the image pyramid and return it as a Numpy array.

        Args:
            level (int): pyramid level to read
            as_type: type of the pixels (default numpy.uint8)

        Returns:
            a numpy.ndarray
        """
        if level < 0 or level >= self.level_count:
            raise RuntimeError("requested level does not exist")

        return self.get_region_px(0, 0, self.widths[level], self.heights[level], level, as_type)

    def get_polygonal_region_px(self, contour: shg.Polygon, level: int,
                                border: int = 0, as_type=np.uint8) -> np.ndarray:
        """Returns a rectangular view of the image source that minimally covers a closed
        contour (polygon). All pixels outside the contour are set to 0.

        Args:
            contour (shapely.geometry.Polygon): a closed polygonal line given in
                terms of its vertices. The contour's coordinates are supposed to be
                precomputed and to be represented in pixel units at the desired level.
            level (int): image pyramid level
            border (int): if > 0, take this many extra pixels in the rectangular
                region (up to the limits on the image size)
            as_type: pixel type for the returned image (array)

        Returns:
            a numpy.ndarray
        """
        x0, y0, x1, y1 = [int(_z) for _z in contour.bounds]
        x0, y0 = max(0, x0 - border), max(0, y0 - border)
        x1, y1 = min(x1 + border, self.extent(level)[0]), \
            min(y1 + border, self.extent(level)[1])
        # Shift the annotation such that (0,0) will correspond to (x0, y0)
        contour = sha.translate(contour, -x0, -y0)

        # Read the corresponding region
        img = self.get_region_px(x0, y0, x1 - x0, y1 - y0, level, as_type=as_type)

        # mask out the points outside the contour
        for i in np.arange(img.shape[0]):
            # line mask
            lm = np.zeros((img.shape[1], img.shape[2]), dtype=img.dtype)
            j = [_j for _j in np.arange(img.shape[1]) if shg.Point(_j, i).within(contour)]
            lm[j,] = 1
            img[i,] = img[i,] * lm

        return img

##