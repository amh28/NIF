# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import numpy as np
import numpy.ma as ma
import math
import scipy.stats.mstats as mstats
import scipy.ndimage as ndimage

from niftynet.utilities.util_common import MorphologyOps, CacheFunctionOutput


class RegionProperties(object):
    def __init__(self, seg, img, measures,
                 num_neighbors=6, threshold=0, pixdim=(1, 1, 1)):

        self.seg = seg
        self.bin = 100
        self.mul = 100
        self.trans = 0
        self.img = img
        self.img_channels = self.img.shape[4] if img.ndim >= 4 else 1
        img_id = range(0, self.img_channels)
        self.neigh = num_neighbors
        self.harilick_m = np.atleast_2d(self.harilick_matrix())
        self.m_dict = {
            'centre of mass': (self.centre_of_mass, ['CoMx',
                                                     'CoMy',
                                                     'CoMz']),
            'volume': (self.volume,
                       ['NVoxels', 'NVoxelsBinary', 'Vol', 'VolBinary']),
            'surface': (self.surface, ['NSurface',
                                       'NSurfaceBinary',
                                       'SurfaceVol',
                                       'SurfaceVolBinary']),
            'surface volume ratio': (self.sav, ['SAVNumb',
                                                'SAVNumBinary',
                                                'SAV',
                                                'SAVBinary']),
            'compactness': (self.compactness, ['CompactNumb',
                                               'CompactNumbBinary',
                                               'Compactness',
                                               'CompactnessBinary']),
            'mean': (self.mean_, ['Mean_%d' % i for i in img_id]),
            'weighted_mean': (self.weighted_mean_,
                              ['Weighted_mean_%d' % i for i in img_id]),
            'median': (self.median_, ['Median_%d' % i for i in img_id]),
            'skewness': (self.skewness_, ['Skewness_%d' % i for i in img_id]),
            'kurtosis': (self.kurtosis_, ['Kurtosis_%d' % i for i in img_id]),
            'min': (self.min_, ['Min_%d' % i for i in img_id]),
            'max': (self.max_, ['Max_%d' % i for i in img_id]),
            'quantile_25': (self.quantile_25,
                            ['P25_%d' % i for i in img_id]),
            'quantile_50': (self.median_,
                            ['P50_%d' % i for i in img_id]),
            'quantile_75': (self.quantile_75,
                            ['P75_%d' % i for i in img_id]),
            'std': (self.std_, ['STD_%d' % i for i in img_id]),
            'asm': (self.call_asm,
                    ['asm%d' % i for i in
                     img_id]),

            'contrast': (self.call_contrast, ['contrast%d' % i for i in
                                              img_id]),
            'correlation': (self.call_correlation, ['correlation%d' % i
                                                    for i
                                                    in
                                                    img_id]),
            'sumsquare': (self.call_sum_square, ['sumsquare%d' % i for i in
                                                 img_id]),
            'sum_average': (self.call_sum_average, ['sum_average%d' % i
                                                    for i in
                                                    img_id]),
            'idifferentmomment': (self.call_idifferent_moment,
                                  ['idifferentmomment%d' % i for i in
                                   img_id]),
            'sumentropy': (self.call_sum_entropy, ['sumentropy%d' % i for i in
                                                   img_id]),
            'entropy': (self.call_entropy, ['entropy%d' % i for i in
                                            img_id]),
            'differencevariance': (self.call_difference_variance,
                                   ['differencevariance%d' % i for i in
                                    img_id]),
            'differenceentropy': (self.call_difference_entropy,
                                  ['differenceentropy%d'
                                   % i for i in
                                   img_id]),
            'sumvariance': (self.call_sum_variance, ['sumvariance%d' % i for i
                                                     in
                                                     img_id]),
            'imc1': (self.call_imc1, ['imc1%d' % i for i in img_id]),
            'imc2': (self.call_imc2, ['imc2%d' % i for i in img_id])

        }
        self.measures = measures

        self.pixdim = pixdim
        self.threshold = threshold
        if self.seg is not None:
            self.masked_img, self.masked_seg = self.__compute_mask()
        self.vol_vox = np.prod(pixdim)

    def __compute_mask(self):
        # TODO: check whether this works for probabilities type_str
        foreground_selector = np.where((self.seg > 0).reshape(-1))[0]
        probs = self.seg.reshape(-1)[foreground_selector]
        regions = np.zeros((foreground_selector.shape[0], self.img_channels))
        for i in np.arange(self.img_channels):
            regions[:, i] = self.img[..., 0, i].reshape(-1)[foreground_selector]
        return regions, probs

    def centre_of_mass(self):
        return np.mean(np.argwhere(self.seg > self.threshold), 0)

    @CacheFunctionOutput
    def volume(self):
        numb_seg = np.sum(self.seg)
        numb_seg_bin = np.sum(self.seg > 0)
        return numb_seg, numb_seg_bin, \
               numb_seg * self.vol_vox, numb_seg_bin * self.vol_vox

    @CacheFunctionOutput
    def surface(self):
        border_seg = MorphologyOps(self.seg, self.neigh).border_map()
        numb_border_seg_bin = np.sum(border_seg > 0)
        numb_border_seg = np.sum(border_seg)
        return numb_border_seg, numb_border_seg_bin, \
               numb_border_seg * self.vol_vox, numb_border_seg_bin * self.vol_vox

    def glcm(self):
        shifts = [[0, 0, 0],
                  [1, 0, 0],
                  [-1, 0, 0],
                  [0, 1, 0],
                  [0, -1, 0],
                  [0, 0, 1],
                  [0, 0, -1],
                  [1, 1, 0],
                  [-1, -1, 0],
                  [-1, 1, 0],
                  [1, -1, 0],
                  [1, 1, 0],
                  [0, -1, -1],
                  [0, -1, 1],
                  [0, 1, -1],
                  [1, 0, 1],
                  [-1, 0, -1],
                  [-1, 0, 1],
                  [1, 0, -1],
                  [1, 1, 1],
                  [-1, 1, -1],
                  [-1, 1, 1],
                  [1, 1, -1],
                  [1, -1, 1],
                  [-1, -1, -1],
                  [-1, -1, 1],
                  [1, -1, -1]]
        bins = np.arange(0, self.bin)
        multi_mod_glcm = []
        if self.seg is None:
            return None
        for m in range(0, self.img.shape[4]):
            shifted_image = []
            for n in range(0, self.neigh + 1):

                new_img = self.seg * self.img[:, :, :, 0, m]
                print(np.max(self.img), 'is max img')
                new_img = ndimage.shift(new_img, shifts[n], order=0)
                print(np.max(self.seg), 'is max shift')
                if np.count_nonzero(new_img) > 0:
                    flattened_new = new_img.flatten()
                    flattened_seg = self.seg.flatten()
                    affine = np.round(flattened_new * self.mul + self.trans)
                    # select = [round(flattened_new[i] * self.mul+self.trans) for i in
                    #                 range(0, new_img.size) if
                    #           flattened_seg[i]>0]

                    select_new = np.digitize(affine[flattened_seg == 1], bins)
                    select_new[select_new >= self.bin] = self.bin - 1
                    print(np.max(select_new), ' is max bin', np.max(affine))
                    shifted_image.append(select_new)
            glcm = np.zeros([self.bin, self.bin, self.neigh])
            for n in range(0, self.neigh):
                for i in range(0, shifted_image[0].size):
                    glcm[shifted_image[0][i], shifted_image[n + 1][i], n] += 1
                glcm[:, :, n] = glcm[:, :, n] / np.sum(glcm[:, :, n])
            multi_mod_glcm.append(glcm)
        return multi_mod_glcm

    def harilick_matrix(self):
        multi_mod_glcm = self.glcm()
        matrix_harilick = np.zeros([13, self.neigh, self.img_channels])
        if multi_mod_glcm is None:
            return np.average(matrix_harilick, axis=1)

        for i in range(0, self.img_channels):
            for j in range(0, self.neigh):
                matrix = multi_mod_glcm[i][..., j]
                harilick_vector = self.harilick(matrix)
                for index, elem in enumerate(harilick_vector):
                    matrix_harilick[index, j, i] = elem
        return np.average(matrix_harilick, axis=1)

    def call_asm(self):
        return self.harilick_m[0, :]

    def call_contrast(self):
        return self.harilick_m[1, :]

    def call_correlation(self):
        return self.harilick_m[2, :]

    def call_sum_square(self):
        return self.harilick_m[3, :]

    def call_sum_average(self):
        return self.harilick_m[4, :]

    def call_idifferent_moment(self):
        return self.harilick_m[5, :]

    def call_sum_entropy(self):
        return self.harilick_m[6, :]

    def call_entropy(self):
        return self.harilick_m[7, :]

    def call_difference_variance(self):
        return self.harilick_m[8, :]

    def call_difference_entropy(self):
        return self.harilick_m[9, :]

    def call_sum_variance(self):
        return self.harilick_m[10, :]

    def call_imc1(self):
        return self.harilick_m[11, :]

    def call_imc2(self):
        return self.harilick_m[12, :]

    def harilick(self, matrix):
        vector_harilick = []
        asm = self.angular_second_moment(matrix)
        contrast = self.contrast(matrix)
        correlation = self.correlation(matrix)
        sum_square = self.sum_square_variance(matrix)
        sum_average = self.sum_average(matrix)
        idifferentmomment = self.inverse_difference_moment(matrix)
        sum_entropy = self.sum_entropy(matrix)
        entropy = self.entropy(matrix)
        differencevariance, differenceentropy = \
            self.difference_variance_entropy(matrix)
        sum_variance = self.sum_variance(matrix)
        imc1, imc2 = self.information_measure_correlation(matrix)
        vector_harilick.append(asm)
        vector_harilick.append(contrast)
        vector_harilick.append(correlation)
        vector_harilick.append(sum_square)
        vector_harilick.append(sum_average)
        vector_harilick.append(idifferentmomment)
        vector_harilick.append(sum_entropy)
        vector_harilick.append(entropy)
        vector_harilick.append(differencevariance)
        vector_harilick.append(differenceentropy)
        vector_harilick.append(sum_variance)
        vector_harilick.append(imc1)
        vector_harilick.append(imc2)
        return vector_harilick

    def angular_second_moment(self, matrix):
        asm = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                asm += matrix[i, j] ** 2
        return asm

    def contrast(self, matrix):
        contrast = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[1]):
                contrast += (j - i) ** 2 * matrix[i, j]
        return contrast

    def homogeneity(self, matrix):
        homogeneity = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[1]):
                homogeneity += matrix[i, j] / (1 - abs(i - j))
        return homogeneity

    def energy(self, matrix):
        energy = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                energy += matrix[i, j] ** 2
        return energy

    def entropy(self, matrix):
        entropy = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                if matrix[i, j] > 0:
                    entropy += matrix[i, j] * math.log(matrix[i, j])
        return entropy

    def correlation(self, matrix):
        range_values = np.arange(0, matrix.shape[0])
        matrix_range = np.tile(range_values, [matrix.shape[0], 1])
        mean_matrix = np.sum(matrix_range * matrix, axis=0)
        sd_matrix = np.sqrt(np.sum((matrix_range - np.tile(mean_matrix,
                                                           [matrix.shape[
                                                                0], 1])) ** 2 * matrix, axis=0))
        correlation = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                if sd_matrix[i] > 0 and sd_matrix[j] > 0:
                    correlation += (i * j * matrix[i, j] - mean_matrix[i] * mean_matrix[
                        j]) / (sd_matrix[i] * sd_matrix[j])
        return correlation

    def inverse_difference_moment(self, matrix):
        idm = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                idm += 1.0 / (1 + (i - j) ** 2) * matrix[i, j]
        return idm

    def sum_average(self, matrix):
        sa = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                sa += (i + j) * matrix[i, j]
        return sa

    def sum_entropy(self, matrix):
        se = 0
        matrix_bis = np.zeros([2 * matrix.shape[0]])
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                matrix_bis[i + j] += matrix[i, j]
        for v in matrix_bis:
            if v > 0:
                se += v * math.log(v)
        return se

    def sum_variance(self, matrix):
        sv = 0
        se = self.sum_entropy(matrix)
        matrix_bis = np.zeros([2 * matrix.shape[0]])
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                matrix_bis[i + j] += matrix[i, j]
        for i in range(0, matrix_bis.size):
            sv += (i - se) ** 2 * matrix_bis[i]
        return sv

    def difference_variance_entropy(self, matrix):
        dv = 0
        de = 0
        matrix_bis = np.zeros([matrix.shape[0]])
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                matrix_bis[abs(i - j)] += matrix[i, j]
        for i in range(0, matrix.shape[0]):
            dv += matrix_bis[i] * i ** 2
            if matrix_bis[i] > 0:
                de -= math.log(matrix_bis[i]) * matrix_bis[i]
        return dv, de

    def information_measure_correlation(self, matrix):
        hxy = self.entropy(matrix)
        sum_row = np.sum(matrix, axis=0)
        hxy_1 = 0
        hxy_2 = 0
        hx = 0
        for i in range(0, matrix.shape[0]):
            hx -= sum_row[i] * math.log(sum_row[i] + 0.001)
            for j in range(0, matrix.shape[0]):
                hxy_1 -= matrix[i, j] * math.log(sum_row[i] * sum_row[j] + 0.001)
                hxy_2 -= sum_row[i] * sum_row[j] * math.log(sum_row[i] *
                                                            sum_row[j] + 0.001)
        ic_1 = (hxy - hxy_1) / (hx)
        if hxy == 0:
            ic_2 = 0
        else:
            ic_2 = math.sqrt(1 - math.exp(-2 * (hxy_2 - hxy)))
        return ic_1, ic_2

    def sum_square_variance(self, matrix):
        ssv = 0
        range_values = np.arange(0, matrix.shape[0])
        matrix_range = np.tile(range_values, [matrix.shape[0], 1])
        mean = np.average(matrix_range * matrix)
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                ssv += (i - mean) ** 2 * matrix[i, j]
        return ssv

    def sav(self):
        Sn, Snb, Sv, Svb = self.surface()
        Vn, Vnb, Vv, Vvb = self.volume()
        return Sn / Vn, Snb / Vnb, Sv / Vv, Svb / Vvb

    def compactness(self):
        Sn, Snb, Sv, Svb = self.surface()
        Vn, Vnb, Vv, Vvb = self.volume()
        return np.power(Sn, 1.5) / Vn, np.power(Snb, 1.5) / Vnb, \
               np.power(Sv, 1.5) / Vv, np.power(Svb, 1.5) / Vvb

    def min_(self):
        return ma.min(self.masked_img, 0)

    def max_(self):
        return ma.max(self.masked_img, 0)

    def weighted_mean_(self):
        masked_seg = np.tile(self.masked_seg, [self.img_channels, 1]).T
        return ma.average(self.masked_img, axis=0, weights=masked_seg).flatten()

    def mean_(self):
        return ma.mean(self.masked_img, 0)

    def skewness_(self):
        return mstats.skew(self.masked_img, 0)

    def std_(self):
        return ma.std(self.masked_img, 0)

    def kurtosis_(self):
        return mstats.kurtosis(self.masked_img, 0)

    def median_(self):
        return ma.median(self.masked_img, 0)

    def quantile_25(self):
        return mstats.mquantiles(self.masked_img, prob=0.25, axis=0).flatten()

    def quantile_75(self):
        return mstats.mquantiles(self.masked_img, prob=0.75, axis=0).flatten()

    def header_str(self):
        result_str = [j for i in self.measures for j in self.m_dict[i][1]]
        result_str = ',' + ','.join(result_str)
        return result_str

    def to_string(self, fmt='{:4f}'):
        result_str = ""
        for i in self.measures:
            for j in self.m_dict[i][0]():
                try:
                    result_str += ',' + fmt.format(j)
                except ValueError:  # some functions give strings e.g., "--"
                    print(i, j)
                    result_str += ',{}'.format(j)
        return result_str
