import os
import re
from glob import glob
import numpy as np # (1.22.3)
import math
from matplotlib import pyplot as plt # (3.5.1)
from matplotlib import colors # (3.5.1)
import matplotlib.patches as mpatches
import pandas as pd # (1.4.2)
import geopandas as gpd # (0.9.0)
import libpysal as lps # (4.6.2)
from libpysal.weights import WSP
import esda # (2.4.1)
import warnings

warnings.filterwarnings('ignore')


class CohortWideMapping:
    def __init__ (self, dir_files: str) -> None:
        self.directory: str = os.path.abspath(dir_files)
        self.n_files: int = 0
        self.ir_img_array: np.ndarray

        self.save_imgs: str = os.path.join(self.directory, 'imgs', 'cwm')

        filepaths: list[str]
        self.filenames: list[str] = []
            
        # Load all files which are supposed to be included in the grouped significance analysis
        subdir_equ = 'masked_projection_imgs'
        filepaths = glob(os.path.join(dir_files, subdir_equ, '*.npy'))
        self.n_files = len(filepaths)
        self.ir_img_array = np.empty(self.n_files, dtype=object)

        for idx, filepath in enumerate(filepaths):
            filename = os.path.basename(filepath)
            self.filenames.append(filename)
            ir_img: np.ndarray = np.load(filepath)
            self.ir_img_array[idx] = ir_img
            # self.ir_img_array.append(ir_img)
        print(f"Loaded files are: {', '.join(self.filenames)}")

    def combine_datasets(self, num_per_col: int) -> None:
        """
        Combine single datasets to superimage

        Parameters:
            num_per_col(int): for equidistant, number of images per column
        """

        # Group individual masked projection images
        self.num_per_col = num_per_col

        rowdims = []
        for i in range(len(self.ir_img_array)):
            rowdim = len(self.ir_img_array[i])
            rowdims.append(rowdim)
        max_row_dim = np.amax(rowdims)
        coldims = []
        for i in range(len(self.ir_img_array)):
            coldim = len(self.ir_img_array[i][0])
            coldims.append(coldim)
        max_col_dim = np.amax(coldims)
        maximal = np.amax([max_col_dim, max_row_dim]) + 1

       # Fill to maximal in squares
        filled_images = []

        for img in self.ir_img_array:
            # Get sheet according to wavelength
            # indexWavelength = np.where(dataWavelengths == wavelength2plot)
            data_set = img
            #img_at_wavelength = data_set[:,:,indexWavelength[0][0]]
            img_at_wavelength = data_set[:,:]
            # Put together 0-2D image and replace left upper corner with real data pixels
            size_arr = np.zeros((maximal,maximal))
            r_len = len(img)
            c_len = len(img[0])
            size_arr[0:r_len, 0:c_len] = img_at_wavelength
            filled_images.append(size_arr)
        new = np.hstack(filled_images)
        
        # Fill with empty fields if not fitting in raster
        if len(self.ir_img_array)/self.num_per_col == True:
            stack_ready2D = new
        else:
            num_missing_raster_fields = int(np.ceil(len(self.ir_img_array)/self.num_per_col)*self.num_per_col - len(self.ir_img_array))
            filler = np.zeros((maximal, maximal*num_missing_raster_fields))
            stack_ready2D = np.hstack((new,filler))
        
        # Split in individual rows
        rows2stack = []
        for i in range(int(len(stack_ready2D[0])/maximal/self.num_per_col)):
            row = stack_ready2D[:, (i*maximal*self.num_per_col):((i+1)*maximal*self.num_per_col)]
            rows2stack.append(row)
        # Stack individual rows together to raster
        self.super_img = np.vstack(rows2stack)

        fig, ax = plt.subplots()
        im = ax.imshow(self.super_img, cmap = "magma")
        im.set_clim(np.amin(self.super_img[self.super_img != 0]),np.amax(self.super_img[self.super_img != 0]))
        cbar = fig.colorbar(im, ax=ax, fraction=0.01, pad=0.04)
        cbar.ax.ticklabel_format(useOffset=False, style='plain')
        plt.show()


    def format_for_propability_mapping(self) -> None:
        # Format grouped data set in preparation of grouped significance mapping
        # Construct the goe dataframe
        for_calculation = self.super_img
        self.width = for_calculation.shape[1]
        self.height = for_calculation.shape[0]
        # size = for_calculation.size
        self.super_mask = np.where(self.super_img == 0, 0, 1)
        size_tissue = np.count_nonzero(self.super_mask, axis=None)
        # Describe part1
        one = 'POINT ('
        part1 = [one]*size_tissue
        # Describe part2 --> actual coordinates of each pixel
        part2 = []
        for i in range(self.height):
            h = self.height - i
            part = []
            for n in range(self.width):
                string = str(n+1) + ' ' + str(h)
                part.append(string)
            part2.append(part)
        part2 = [j for sub in part2 for j in sub] # part2 are all pixels recorded (including background)
        mask_flat = self.super_mask.flatten()
        indiv_mask = np.where(mask_flat == 1)[0]
        coord_tissue_only = [part2[i] for i in indiv_mask]
        # Describe part3
        three = ')'
        part3 = [three]*size_tissue
        # Bring everything together to final list of strings which can be used as geometry data
        coord = [m+str(n)+str(o) for m,n,o in zip(part1, coord_tissue_only, part3)]
        # extract intensities which are not background
        ind_tissue_intens = np.where(mask_flat == 1)
        intens_data_w_mask = for_calculation.flatten()
        tissue_intens = intens_data_w_mask[ind_tissue_intens]
        # Create data frame out of data
        frame = {'intensities': tissue_intens, 'wkt': coord}
        dataframe = pd.DataFrame(frame)
        geoseries = gpd.GeoSeries.from_wkt(dataframe['wkt'])
        self.geodataframe = gpd.GeoDataFrame(dataframe, geometry=geoseries, crs="EPSG:4326")
        print(self.geodataframe.head())

    def perform_cohorwide_prob_mapping(self, confidence_level) -> None:
        """

        """
        # Calculate Spatial attribute
        wqnew: WSP = lps.weights.Queen.from_dataframe(self.geodataframe)
        # and normalize
        wqnew.transform = 'r'
        # Extract intensity attribute
        ynew = self.geodataframe['intensities']
        # Calculate lag data
        # ylagnew = lps.weights.lag_spatial(wqnew, ynew)
        np.random.seed(12345)
        # Assign each data point to one of the quadrants
        linew = esda.moran.Moran_Local(ynew, wqnew)
        linew.q
        # Extract the significant values by choosing a p-value
        signew = linew.p_sim < confidence_level
        highspotnew = signew * linew.q==1
        copy = self.super_mask.flatten()
        bgcopy = np.where(copy == 0, -10, copy)
        indxa = np.where(bgcopy == 1) 
        bgcopy[indxa] = highspotnew
        self.highplot = bgcopy.reshape(self.height, self.width) 
        # Introduce suiting color map
        self.cmap_high = colors.ListedColormap(['white', 'lightgrey', 'darkred'])
        boundaries = [-15, -5, 0.5, 1.5]
        self.norm = colors.BoundaryNorm(boundaries, self.cmap_high.N, clip=True)
        fig, ax = plt.subplots()
        ax.imshow(self.highplot, cmap=self.cmap_high, norm=self.norm)
        plt.show()


    def split_into_individual_maps(self, indiv_example: int) -> None:
        # Separate results of individual samples retrieved from grouped significance map
        self.indiv_example: int = indiv_example
        # Describe grouping of individual images
        num_img_cols = 3  # Number of sample columns
        num_img_rows = 1  # Number of sample rows
        sum_stack_height = len(self.highplot) # --> # of rows 
        # sumStackWidth = len(self.highplot[0])  # --> # of columns
        square_size_individuals = int(sum_stack_height / num_img_rows)
        # Repack in one row to extract individuals of the same size
        one_row_img = []
        for i in range(num_img_rows):
            row_img = self.highplot[0 + i*square_size_individuals:square_size_individuals \
                              + i*square_size_individuals, 0:square_size_individuals*num_img_cols]
            one_row_img.append(row_img)
        one_row_img = np.hstack(one_row_img)
        self.individual_hotspot_maps = []
        for i in range(len(self.ir_img_array)):  # = number of image stacks = num_img_cols*num_img_rows
        # for ir_img in self.ir_img_array
            sliced_img = one_row_img[0:square_size_individuals, 0+i*square_size_individuals \
                                  :square_size_individuals+i*square_size_individuals]
            sliced_img = sliced_img[:, :, np.newaxis]
            self.individual_hotspot_maps.append(sliced_img)
        fig, ax = plt.subplots()
        ax.imshow(self.individual_hotspot_maps[indiv_example], cmap=self.cmap_high, norm=self.norm)
        red_patch = mpatches.Patch(color='darkred', label='significant pixels')
        gray_patch = mpatches.Patch(color='lightgray', label='non-significant pixels')
        ax.legend(handles=[red_patch, gray_patch], loc='center left')
        plt.show()


    def save_prob_maps(self) -> None:
        """

        """

        parent_dir = os.path.basename(self.directory)
        # file_substr_list: list[str] = [filename.split('_')[-1].split('.')[0] for filename in self.filenames]
        file_substr_list: list[str] = [re.split(r'[_.]', filename)[-2] for filename in self.filenames]

        save_subdir: str = 'probability_maps/cohort_wide_mapping'
        save_subname: str = 'HotSpotMap_cohortwide_tumor_1proz_'
        savedir_equ: str = os.path.join(self.directory, save_subdir)
        os.makedirs(savedir_equ, exist_ok=True)
        try:
            for idx, file_substr in enumerate(file_substr_list):
                np.save(os.path.join(savedir_equ, save_subname + file_substr + '.npy'), self.individual_hotspot_maps[idx])
            print(f"Cohort-wide mapping data saved in '{parent_dir}/{save_subdir}'.") 
        except AttributeError:
            raise AttributeError(f"Previous module(s) not excecuted to create data to sae.")

