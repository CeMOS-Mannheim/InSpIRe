from collections import Counter
from glob import glob
import os
from random import sample
import re
from typing import Literal

import cv2 # (4.5.5)
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt # (3.5.1)
import numpy as np # (1.22.3)
import pandas as pd
from PIL import Image # (9.1.0)
from PIL.ImageFile import ImageFile
import seaborn as sns # (0.11.2)
from scipy import ndimage # (1.8.0)
from scipy.ndimage import rotate # (1.8.0)
import skimage.exposure as exposure # (0.19.2)
from skimage.morphology import binary_erosion  # (0.19.2)
from sklearn.cluster import KMeans # (1.0.2)
from sklearn.ensemble import RandomForestClassifier # (1.0.2)
from sklearn.model_selection import train_test_split # (1.0.2)
from specio_py310 import specread
from specio_py310.core import Spectrum

from ._coregistration_workflow import register_image_elastix, transformix_image

import warnings
warnings.filterwarnings('ignore')


def bg_norm(data: np.ndarray) -> np.ndarray:
    x_shape, y_shape, z_shape = map(int, data.shape)
    norm_img_cube: np.ndarray = np.zeros([x_shape, y_shape, z_shape])
    for z_element in range(z_shape):
        img: np.ndarray = np.array(data[..., z_element])
        bg_left: np.ndarray = img[:, 0]
        bg_right: np.ndarray = img[:,-1]
        average: np.floating = np.mean((bg_left, bg_right))
        factor: np.floating = 1 / average
        bg_norm_img: np.ndarray = img * factor
        # Save normalized img
        norm_img_cube[..., z_element] = bg_norm_img

    return norm_img_cube


def derivative(data: np.ndarray, dx: float, order: int) -> np.ndarray:
    size_x: np.ndarray = np.shape(data)[1]
    size_y: np.ndarray = np.shape(data)[0]
    size_z: np.ndarray = np.shape(data)[2]
    data = data.reshape(size_y*size_x, size_z)
    data = np.diff(data, order)/dx**order
    data = data.reshape(size_y, size_x, size_z - order)
    return data


def vector_norm(data: np.ndarray) -> np.ndarray:
    size_x: np.ndarray = np.shape(data)[1]
    size_y: np.ndarray = np.shape(data)[0]
    size_z: np.ndarray = np.shape(data)[2]
    data = data.reshape(size_y*size_x, size_z)
    squared: np.ndarray = np.square(data)
    spec_sum: np.ndarray = np.sum(squared, axis = 1)
    spec_sqrt: np.ndarray = np.sqrt(spec_sum)
    data_vn: np.ndarray = data / spec_sqrt.reshape(len(spec_sqrt), 1)
    data_vn = data.reshape(size_y, size_x, size_z)
    return data_vn


class PreProcessingInfrared:
    def __init__(self, directory: str) -> None:
        """
        Loading and preprocessing of infrared data


        Parameter:
            datadir (str): Directory where infrared data is stored
        """
        self.directory: str = os.path.abspath(directory)
        self.file_paths: list[str] = glob(os.path.join(self.directory, 'MIRimaging_raw_data', '*'))
        self.filenames: list[str] = [os.path.basename(file) for file in self.file_paths]
        self.information: str = os.path.join(self.directory, 'feature_selection_prior_knowledge')

        self.ir_spectra: Spectrum
        self.indiv_2darrays_preprocessed: list[np.ndarray] = []
        self.sample_number: int
        self.second_deriv_wavenumbers: np.ndarray
        self.img_plot: np.ndarray
        self.prior_knowledge: Literal['features', 'rois']
        # self.tumormask: np.ndarray
        # self.non_tumormaskf: np.ndarray
        self.image_grey: np.ndarray
        self.angle: int
        self.ir_img: np.ndarray
        self.ymid_moving: float
        self.xmid_moving: float
        self.yslice: int
        self.xslice: int
        self.map000: np.ndarray
        # self.tumor_rot: np.ndarray
        # self.non_tumor_rot: np.ndarray

        self.subset: int
        self.projection_imgs: list[np.ndarray] = []

        self.feature_list: list[int]
        self.feature_list_str: list[str]
        self.roi_values: dict[str, np.ndarray] =  {}
        self.predictive_status_list: list[str] =  []
        self.indiv_maskarrays: list[str] = []
        self.masked_project_imgs: list[str] = []
        self.save_imgs: str = os.path.join(self.directory, 'imgs', 'infra')

        numbered_files = [f"{idx}. {name}" for idx, name in enumerate(self.filenames, start=1)]
        print(f"Found {len(self.file_paths)} files: {', '.join(numbered_files)}.")
        

        for filepath in self.file_paths:
            self.ir_spectra = specread(filepath)
            amplitudes_raw: np.ndarray = self.ir_spectra.amplitudes
            meta = self.ir_spectra.meta
            image_dimensions: list[int] = [meta["n_x"], meta["n_y"], meta["n_z"]]
            self.image_stack: np.ndarray = amplitudes_raw.reshape(image_dimensions[1], image_dimensions[0],
                                                image_dimensions[2])
            # Pre-processing -------
            # BG norm
            data: np.ndarray = bg_norm(data = self.image_stack)
            # 2nd deriv
            dx: int = abs(meta["z_delta"])
            data = derivative(data = data, dx = dx, order = 2)
            # Vector normalization
            data = vector_norm(data)
            # Add y-directional offset to avoid negative values
            data = data + 1
            # Save individually in indiv2Darrays_2ndVn
            self.indiv_2darrays_preprocessed.append(data)

    def plot_preprocessed_ir(self, sample_number: int, wavenumber_2plot: int) -> None:
        """
        Evaluating preprocessed data with plot

        Parameter:
            sample_number (int): number of file to plot
            wavenumber_2plot (int): wavenumber to plot
        """
        self.sample_number = sample_number - 1
        # Checking the resulting images
        self.second_deriv_wavenumbers = self.ir_spectra.wavelength[1:-1]
        data_stack_indiv_img = self.indiv_2darrays_preprocessed[self.sample_number]
        index_wavenumber_2plot = np.where(self.second_deriv_wavenumbers == wavenumber_2plot)[0]
        self.img_plot = data_stack_indiv_img[..., index_wavenumber_2plot]
        fig, ax = plt.subplots(figsize=(8,8))
        im = ax.imshow(self.img_plot, cmap="magma")
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        cbar.ax.ticklabel_format(useOffset=False, style='plain')
        ax.axis('off')
        plt.show()


    # def feature_selection(self, prior_knowledge: Literal['features', 'rois'],
    #                       feature_list: list[int] = [], predictive_status: list[Literal['pos', 'neg']] = []) -> None:
    def feature_selection(self, prior_knowledge: Literal['features', 'rois']) -> None:
        """
        feature_list = 964
        predictive_status = pos

        Parameters:
            proir_knowledge(str):
        """
        # Select features which show discrimination between the ROIs
        self.prior_knowledge = prior_knowledge
        if self.prior_knowledge == "features":
            path_features: str = glob(os.path.join(self.information, 'features', '*'))[0]
            features = pd.read_csv(path_features)

            feature_list: list[str] = features.iloc[:, 0].tolist()
            predictive_status: list[str] = features.iloc[:, 1].tolist()
            if predictive_status or feature_list is True:
                try:
                    print(f"\npredictive_status:  {',  '.join(predictive_status)}")
                    # feature_list_str: list[str] = [" " + str(feature) if feature < 1000 else str(feature) for feature in feature_list]
                    print(f"feature_list: {[feature_list]}")

                except AttributeError:
                    raise AttributeError(f"Form of specific feature and predictive_status")

            else: 
                print("Please specify features and predictive status inside the CSV'\
                       for the usage of 'proir_knowledge=features'")
                exit()

            # Wavenumbers associated with nucleic acids were found to show higher absorbance in  
            # ... tumorous regions which includes (964, 1088, 916) cm^(-1) and thus 
            # ... could directly be used for subsequent projection
            # self.feature_list = [964, 1088, 916]
            # self.predictive_status_list = ['pos', 'pos', 'pos']
            #NOTE: version with user input
            # print("Input wavenumbers as feature list like '964, 1088, 916'")
            # features: str = input()
            # feature_list: list[str] = features.split(', ')
            # self.feature_list = list(map(int, feature_list))
            # #TODO: Add check for predictive status
            # print("Input predictive status like 'pos, pos, neg'")
            # predictive_status: str = input()
            # self.predictive_status_list = predictive_status.split(', ')
            # print(f"Jump to method 'calc_projection_image'.")

        # Select features which show discrimination between the ROIs
        if prior_knowledge == "rois":
            # Load the information on the ROIs
            path_image_rois = glob(os.path.join(self.information, 'rois', '*'))[0]
            im: ImageFile = Image.open(path_image_rois)
            image = np.array(im)  
            # Extract ROIs based on their color assignment
            b, g, r = cv2.split(image)

            # TUMOR (green)
            col = [1,217,15] # center_color
            prec = 100 # precision
            tumoranno = ((b >= col[0]-prec) & (b <= col[0]+prec)) & ((g >= col[1]-prec) \
                         & (g <= col[1]+prec)) & ((r >= col[2]-prec) & (r <= col[2]+prec))
            # Fill wholes of annotated structures
            tumormask = ndimage.binary_fill_holes(tumoranno[:,:]).astype(int)
            kernel = np.ones((5,5))
            tumormask = binary_erosion(tumormask, footprint=kernel, out=None)

            # NON-Tumor ROIs

            # Tumor lumen / starting necrosis
            col = [244, 177, 131]
            prec = 10
            lumenanno = ((b >= col[0]-prec) & (b <= col[0]+prec)) & ((g >= col[1]-prec) \
                         & (g <= col[1]+prec)) & ((r >= col[2]-prec) & (r <= col[2]+prec))
            lumenmask = ndimage.binary_fill_holes(lumenanno[:,:]).astype(int)
            lumenmask = binary_erosion(lumenmask, footprint=kernel, out=None)

            # Connective tissue (yellow)
            col = [246,249,20]
            prec = 100
            connectiveanno = ((b >= col[0]-prec) & (b <= col[0]+prec)) & ((g >= col[1]-prec) \
                              & (g <= col[1]+prec)) & ((r >= col[2]-prec) & (r <= col[2]+prec))
            connectivemask = ndimage.binary_fill_holes(connectiveanno[:,:]).astype(int)
            connectivemask = binary_erosion(connectivemask, footprint=kernel, out=None) 

            # Necrosis (black)
            col = [0,0,0]
            prec = 10
            necrosisanno = ((b >= col[0]-prec) & (b <= col[0]+prec)) & ((g >= col[1]-prec) \
                            & (g <= col[1]+prec)) & ((r >= col[2]-prec) & (r <= col[2]+prec))
            necrosismask = ndimage.binary_fill_holes(necrosisanno[:,:]).astype(int)
            necrosismask = binary_erosion(necrosismask, footprint=kernel, out=None)
            
            # Liver parenchyma
            col = [253,102,119]
            prec = 10
            parenchymanno = ((b >= col[0]-prec) & (b <= col[0]+prec)) & ((g >= col[1]-prec) \
                             & (g <= col[1]+prec)) & ((r >= col[2]-prec) & (r <= col[2]+prec))
            parenchymmask = ndimage.binary_fill_holes(parenchymanno[:,:]).astype(int)
            parenchymmask = binary_erosion(parenchymmask, footprint=kernel, out=None)

            # Inflammation (red)
            col = [227,6,19]
            prec = 10
            inflamanno = ((b >= col[0]-prec) & (b <= col[0]+prec)) & ((g >= col[1]-prec) \
                          & (g <= col[1]+prec)) & ((r >= col[2]-prec) & (r <= col[2]+prec))
            inflammask = ndimage.binary_fill_holes(inflamanno[:,:]).astype(int)
            inflammask = binary_erosion(inflammask, footprint=kernel, out=None)

            # For simplification in this exemplary code, NON-tumorous ROIs are combined into one class
            tumormask = np.ma.masked_where(tumormask == 0, tumormask)
            non_tumormask = np.where((necrosismask + connectivemask + inflammask 
                                    + parenchymmask + lumenmask) >= 1, 1, 0)
            non_tumormask = np.ma.masked_where(non_tumormask == 0, non_tumormask)

            self.masks = {'tumor': tumormask, 'non_tumor': non_tumormask, 'lumen': lumenmask, 'connective': connectivemask, 'necrosis': necrosismask, 'parenchy': parenchymmask, 'inflam': inflammask}

            # Select features which show discrimination between the ROIs
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
            ax1.imshow(image)
            ax1.set_title("Annotated H&E image")
            # ---
            # Use grayscale for reference 
            self.image_grey = im.convert('L')
            self.image_grey = np.array(self.image_grey)
            ax2.imshow(self.image_grey, cmap='Greys', vmin=0, vmax=400)
            ax2.imshow(self.masks['tumor'], cmap='Greens', vmin=0, vmax=1) 
            ax2.imshow(self.masks['non_tumor'], cmap='Blues', vmin=0, vmax=1)
            green_patch = mpatches.Patch(color='darkgreen', label='Tumor')
            blue_patch = mpatches.Patch(color='darkblue', label='Non-tumor')
            ax2.legend(handles=[green_patch, blue_patch], loc="center left", bbox_to_anchor=(1, 0.5) ) 
            ax2.set_title("Extracted annotations")
            plt.show()


    def co_registration_matrix(self, mirror: bool, angle: int) -> None:
        self.angle = angle
        self.ir_img = self.img_plot[:,:,0]
        # Original  ----------------------------------------------------------------------------
        cutimage_grey: np.ndarray = self.image_grey
        # Mirror -------------------------------------------------------------------------------
        # Mirrowing required: ---------------
        # Flip YES
        if mirror:
            cutimage_grey = np.flip(cutimage_grey, 1)
        # Flip NO
        # -----------------------------------
        # Rotate -------------------------------------------------------------------------------
        cutimage_grey = rotate(cutimage_grey, self.angle) 
        # Slice --------------------------------------------------------------------------------
        # Get size of fixed image (H&E) to adapt moving img (IR img)
        yfixed = len(self.ir_img)
        xfixed = len(self.ir_img[0])
        # Get mid-point of moving image to crop from left and right / top and bottom
        self.ymid_moving = len(cutimage_grey) / 2 + 10
        self.xmid_moving = len(cutimage_grey[0]) / 2  +  0
        # Adapt differences in resolution between IR img (25 µm) and H&E img (2.5 µm)
        self.yslice = round(yfixed*2.25)
        self.xslice = round(xfixed*2.25)
        # Slice moving image to the same proportions as fixed image
        sliced_moving: np.ndarray = cutimage_grey[round(self.ymid_moving-(self.yslice/2))
                                                  :round(self.ymid_moving+(self.yslice/2)), 
                                                  round(self.xmid_moving-(self.xslice/2))
                                                  :round(self.xmid_moving+(self.xslice/2))]

        # Co-register H&E to IR img --------------------------------------
        moved_image, self.map000 = register_image_elastix(fixed = self.ir_img, moving = sliced_moving,
                             method = 'bspline', iterations = 2000, fixed_pixel_size = 25/1000,
                             moving_pixel_size = 13/1000)
        ir_rescaled: np.ndarray = (exposure.rescale_intensity(moved_image, in_range=(0, 255), 
                             out_range=(255, 0))).astype('uint8')
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(ir_rescaled, cmap = "magma")
        ax.imshow(moved_image, cmap = "gist_gray", alpha = 0.8)
        plt.show()

    def co_registraton_transform(self) -> None:
        """
        If the prior knowledge about ROIs is aimed to be used for discriminant feature analysis,
        ... ROIs need to be co-registered with IRI data
        """
        #TODO: Add case for feature list for functions with proir_knowledge

        if self.prior_knowledge == "rois":
            # Transform the binary masks (tumor and non-tumor) the same way to fit the IR image
            # # Tumor ROI ------------------------------------------------
            # self.tumor_rot = rotate(self.tumormask, self.angle) 
            # self.tumor_rot = self.tumor_rot[round(self.ymid_moving-(self.yslice/2)):round(self.ymid_moving+(self.yslice/2)), 
            #                        round(self.xmid_moving-(self.xslice/2)):round(self.xmid_moving+(self.xslice/2))]
            # self.tumor_rot = self.tumor_rot.astype('uint8')
            # self.tumor_rot = transformix_image(image=self.tumor_rot, transform_parameter_map= self.map000,
            #                                          transform_pixel_size=13/1000)
            # self.tumor_rot = np.where(self.tumor_rot> 75, 1, 0)
            # self.tumor_rot = ndimage.binary_closing(self.tumor_rot,
            #                                  structure=np.ones((3,3))).astype(int)
            # # NON-tumor ROI ------------------------------------------------
            # self.non_tumor_rot = rotate(self.non_tumormask, self.angle) 
            # self.non_tumor_rot = self.non_tumor_rot[round(self.ymid_moving-(self.yslice/2)):round(self.ymid_moving+(self.yslice/2)), 
            #                  round(self.xmid_moving-(self.xslice/2)):round(self.xmid_moving+(self.xslice/2))]
            # self.non_tumor_rot = self.non_tumor_rot.astype('uint8')
            # self.non_tumor_rot = transformix_image(image=self.non_tumor_rot, transform_parameter_map=self.map000,
            #                                       transform_pixel_size=13/1000)
            # self.non_tumor_rot = np.where(self.non_tumor_rot> 75, 1, 0)
            # self.non_tumor_rot = ndimage.binary_closing(self.non_tumor_rot, structure=np.ones((3,3))).astype(int)
            #
            # self.all_mask = rotate(self.all_mask, self.angle) 
            # self.all_mask = self.all_mask[round(self.ymid_moving-(self.yslice/2)):round(self.ymid_moving+(self.yslice/2)), 
            #                  round(self.xmid_moving-(self.xslice/2)):round(self.xmid_moving+(self.xslice/2))]
            # self.all_mask = self.all_mask.astype('uint8')
            # self.all_mask = transformix_image(image=self.all_mask, transform_parameter_map=self.map000,
            #                                       transform_pixel_size=13/1000)
            # self.all_mask = np.where(self.all_mask> 75, 1, 0)
            # self.all_mask = ndimage.binary_closing(self.all_mask, structure=np.ones((3,3))).astype(int)
            #
            # # Show results
            # fig, ax = plt.subplots(figsize=(10,5))
            # ax.imshow(self.ir_img, cmap="magma")
            # ax.contour(self.all_mask['tumor'], colors='lime', linewidths=0.3 )  
            # ax.contour(self.all_mask['non_tumor'], colors='blue', linewidths=0.3)
            # proxy = [Line2D([0], [0], color='lime'), Line2D([0], [0], color='blue')]
            # labels = ['tumor', 'non-tumor']
            # ax.legend(proxy, labels, loc='center left', bbox_to_anchor=(1, 0.5))
            # fig.savefig(os.path.join(self.save_imgs, 'coregistration_transform.pdf'), bbox_inches='tight')
            # fig.show()

            for ttype in self.masks:
                self.masks[ttype] = rotate(self.masks[ttype], self.angle)
                self.masks[ttype] = self.masks[ttype][round(self.ymid_moving-(self.yslice/2)):round(self.ymid_moving+(self.yslice/2)), 
                                             round(self.xmid_moving-(self.xslice/2)):round(self.xmid_moving+(self.xslice/2))]
                self.masks[ttype] = self.masks[ttype].astype('uint8')
                self.masks[ttype] = transformix_image(image=self.masks[ttype], transform_parameter_map=self.map000,
                                                                  transform_pixel_size=13/1000)
                self.masks[ttype] = np.where(self.masks[ttype]> 75, 1, 0)
                self.masks[ttype] = ndimage.binary_closing(self.masks[ttype], structure=np.ones((3,3))).astype(int)
                
            # Show results
            fig, ax = plt.subplots(figsize=(10,5))
            ax.imshow(self.ir_img, cmap="magma")
            ax.contour(self.masks['tumor'], colors='lime', linewidths=0.3 )  
            ax.contour(self.masks['non_tumor'], colors='blue', linewidths=0.3)
            # proxy = [Line2D([0], [0], color='lime'), Line2D([0], [0], color='blue')]
            labels = ['tumor', 'non-tumor']
            # ax.legend(proxy, labels, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()


    def perform_discriminant_analysis(self, subset: int) -> None:
        # Perform discriminant analysis to determine wavenumbers with discriminant power
        self.subset = subset

        list_important_wavenumbers = []
        [width_image, height_image] = self.masks['tumor'].shape
        pixel_category_labels = np.zeros([width_image, height_image], dtype=np.float64)
        for idx, tissue_type in enumerate(self.masks, start=1):
            if idx == 3:  # only for tumor and grouped non-tumor
                break
            category = np.where(self.masks[tissue_type] == 1, idx, 0)
            tissue_random_idx = sample(list(np.where(category.flatten() == idx)[0]), subset)
            tissue_sample = np.zeros((len(category), len(category[0]))).flatten()
            tissue_sample[tissue_random_idx] = idx
            tissue_sample = tissue_sample.reshape((len(category), len(category[0])))
            pixel_category_labels += tissue_sample

        pixel_category_labels = pixel_category_labels.flatten()
        lst_categoriesOI = [1, 2]
        # Retrieve indices of both categories
        idx_categoriesOI = []
        for idx, pixel_category_label in enumerate(pixel_category_labels):
            if pixel_category_label in lst_categoriesOI:
                idx_categoriesOI.append(idx)
        # Assign pixels of IR imaging dataset to these categories
        individual_data_stack = self.indiv_2darrays_preprocessed[self.sample_number] # Get IR image stack
        # Subtract and reorder image stack as required for forest tree
        for _ in range(10):
            list_wavenumber_rows = []
            for i in range(np.shape(individual_data_stack)[2]):
                wavenumber_img = individual_data_stack[:, :, i]
                wavenumber_row = wavenumber_img.flatten()
                list_wavenumber_rows.append(wavenumber_row)
            stacked_wavenumber_rows = np.stack(list_wavenumber_rows)
            # Slice the data stack based on the categories of Interest (classesOI)
            sliced_2d_stacked_wavenumber_rows = stacked_wavenumber_rows[:, idx_categoriesOI]
            sliced_pixel_class_labels = pixel_category_labels[idx_categoriesOI]
            
            # Prepare for discriminant analysis
            x = np.transpose(sliced_2d_stacked_wavenumber_rows)
            y = sliced_pixel_class_labels
            # Perform random forest analysis
            x_train, _, y_train, _ = train_test_split(x, y, stratify=y, random_state=42)
            forest = RandomForestClassifier(random_state=0)
            forest.fit(x_train, y_train)
            importances = forest.feature_importances_
            # Sort importances by biggest value and plot for visual assessment
            row_one: np.ndarray = abs(importances)
            row_two: np.ndarray = self.second_deriv_wavenumbers
            unsorted_arr = np.stack((row_one, row_two))
            # Sort 2D numpy array by 2nd Column
            sorted_arr = unsorted_arr[0,:].argsort()
            sorted_arr = unsorted_arr[:, sorted_arr]
            abs_impo = sorted_arr[0,:]
            wavenumbers = sorted_arr[1,:]
            y_pos: list[int] = []
            # for i in range(len(wavenumbers)):
            for idx, _ in enumerate(wavenumbers):
                wavenumber = str(sorted_arr[1,:][idx])
                y_pos.append(int(float(wavenumber)))
            list_important_wavenumbers.extend(y_pos[-10:])
            y_pos_str: list[str] = list(map(str, y_pos))

        counted_duplicates = Counter(list_important_wavenumbers)
        counted_duplicates = counted_duplicates.most_common(10)
        self.feature_list = [wavenumber for wavenumber, _ in counted_duplicates]
        self.feature_list_str: list[str] = list(map(str, self.feature_list))

        x_min = 812
        x_max = x_min - 50
        fig, ax = plt.subplots(figsize=(20,10))
        ax.bar(y_pos_str[x_max:x_min], abs_impo[x_max:x_min], color = "grey")
        ax.bar(y_pos_str[x_min-10:x_min], abs_impo[x_min-10:x_min], color = "lawngreen")
        ax.invert_xaxis()
        ax.tick_params(axis='x', rotation=90, labelsize=20)
        ax.set_ylabel("Importance", fontsize=20)
        ax.set_xlabel("Wavenumbers (cm^(-1))", fontsize=20)
        plt.show()


    def error_test_wavenumber(self, wavenumber_in_query: int = 0) -> None:
        if wavenumber_in_query not in self.feature_list:
            print(f"Wavenumver for visualization {wavenumber_in_query} not found. Possible wavenumbers are {', '.join(self.feature_list_str)} input a correct wavenumber.")


        for wavenumber in self.feature_list:
            # obtain intensity values at the wavenumber in query
            idx_wavenumber = np.where(self.second_deriv_wavenumbers == wavenumber)
            wavelength2plot = self.second_deriv_wavenumbers[idx_wavenumber]
            second_deriv_wavelengths = self.ir_spectra.wavelength[1:-1]
            data_stack_indiv_img = self.indiv_2darrays_preprocessed[self.sample_number]
            index_wavelength2plot = np.where(second_deriv_wavelengths == wavelength2plot)[0]
            img_plot = data_stack_indiv_img[..., index_wavelength2plot]

            # Mask all intensities which do not belong to the ROI
            # self.roi_values: dict[str, np.ndarray] = {}
            for idx, tissue_type in enumerate(self.masks, start=1):
                masked_roi_ir = img_plot[..., 0] * self.masks[tissue_type]
                roi_values = masked_roi_ir.flatten()[np.where(masked_roi_ir.flatten() != 0)]
                self.roi_values[tissue_type] = roi_values
            list_rois = list(self.roi_values.values())

            # Calculate predictive status based on mean of tumor and non-tumor
            mean_non_tumor = self.roi_values['non_tumor'].mean()
            mean_tumor = self.roi_values['tumor'].mean()
            if mean_tumor > mean_non_tumor:
                predictive_status: str = 'pos'
                self.predictive_status_list.append(predictive_status)
            else:
                predictive_status: str = 'neg'
                self.predictive_status_list.append(predictive_status)   
                
            # Calculate optimal cutoff value from subset
            # ... both data sets need to be of comparable size to determine a optimum
            #TODO: diferent number of subset; fix to global (500) or na?
            num_of_subset = 300
            tumor_subset = sample(list(list_rois[0]), num_of_subset)
            non_tumor_subset = sample(list(list_rois[1]), num_of_subset)
            ymin = np.amin(tumor_subset) + (np.amax(tumor_subset)-np.amin(tumor_subset))*0.15
            ymax = np.amax(tumor_subset) - (np.amax(tumor_subset)-np.amin(tumor_subset))*0.15
            cutoffs = np.arange(ymin, ymax, 0.000005)
            total_num_pixels = len(tumor_subset) + len(non_tumor_subset)
            false_percentages = []

            # Define 
            if predictive_status == 'pos':
                for cutoff in cutoffs:
                    false_tumor = len(np.where(tumor_subset < cutoff)[0])
                    false_non_tumor = len(np.where(non_tumor_subset > cutoff)[0])
                    false_percentage = (false_tumor + false_non_tumor) / total_num_pixels * 100
                    false_percentages.append(false_percentage)
            else:
                for cutoff in cutoffs:
                    false_tumor = len(np.where(tumor_subset > cutoff)[0])
                    false_non_tumor = len(np.where(non_tumor_subset < cutoff)[0])
                    false_percentage = (false_tumor + false_non_tumor) / total_num_pixels * 100
                    false_percentages.append(false_percentage)
                    
            # ... the exact position of that optimum can be calculated by the 1st derivative 
            # ... with its intersec at y = 0
            a, b, c = np.polyfit(cutoffs, false_percentages, 2)
            # The exact position of the minimum lies at:
            intercept = -b / (2*a)
            
            if wavenumber == wavenumber_in_query:
                print(f"Wavenumber in query: {wavenumber}") 
                false_percentages_tissue = {}
                if mean_tumor > mean_non_tumor: 
                    print("(Positive predictive value)")
                    for tissue, roi_values in self.roi_values.items():
                        if tissue == 'tumor':
                            tumor_error = len(np.where(roi_values < intercept)[0])
                            false_tumor = tumor_error / (len(roi_values)) * 100
                            false_percentages_tissue[tissue] = false_tumor
                            print("Tumor error:",round(false_tumor), "%")
                        else:
                            tissue_error = len(np.where(roi_values > intercept)[0])
                            false_tissue = tissue_error / (len(roi_values)) * 100
                            false_percentages_tissue[tissue] = false_tissue
                            print("Tissue error:",round(false_tissue), "%")
                
                elif mean_tumor < mean_non_tumor:
                    print("(Negative predictive value)")
                    for tissue, roi_values in self.roi_values.items():
                        if tissue == 'tumor':
                            tumor_error = len(np.where(roi_values > intercept)[0])
                            false_tumor = tumor_error / (len(roi_values)) * 100
                            false_percentages_tissue[tissue] = false_tumor
                            print("Tumor error:",round(false_tumor), "%")
                        else:
                            tissue_error = len(np.where(roi_values < intercept)[0])
                            false_tissue = tissue_error / (len(roi_values)) * 100
                            false_percentages_tissue[tissue] = false_tissue
                            print(f"{tissue} error:",round(false_tissue), "%")

                interpolation = a*cutoffs**2 + b*cutoffs**1 + c*cutoffs**0
                first_deriviative = 2*a*cutoffs**1 + 1*b*cutoffs**0
                fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(18, 5), gridspec_kw={'width_ratios': [1, 3]})
                # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.7, hspace=0.7)
                plt.subplots_adjust(wspace=0.5)
                ax1.plot(cutoffs, interpolation, color="red", label="Interpolated curve")
                ax1.plot(cutoffs, false_percentages, color="lime", label="Empirical values")
                ax1.vlines(interpolation, 20, 50)
                ax1.set_xlim(ymin, ymax)
                ax1.set_xlabel("Cutoff values")
                # ax1.set_ylabel("Total false assignments (TFA) [%]")
                ax1.set_ylabel("Percentage of false assignments (%)")
                ax1.set_xticklabels(labels = cutoffs, rotation=45, fontsize=7)
                ax1.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
                ax1.legend(loc="upper left")
            
                ax2 = ax1.twinx()
                ax2.plot(cutoffs, first_deriviative, color="red", label="First derivative", linestyle="--")
                ax2.set_xlim(ymin, ymax)
                ax2.hlines(0, ymin, ymax, color="black")
                ax2.set_ylabel("First derivative of TFA")
                ax2.legend(loc="lower right")
            
                sns.violinplot(data=list_rois, density_norm="count", common_norm=True, palette=('lime', 'lightgrey', 'orange', 'yellow', 'black', 'deeppink', 'red'), inner='box', linewidth=0.1,  ax=ax3)
                # ax3.set_xticklabels(['Tumor','Non-tumor (combined)', 'Luminal debris', 'Connective tissue', 'Necrosis', 'Liver parenchyma', 'Inflammation'], rotation=45, fontsize=10)
                ax3.set_xticklabels(['Tumor','Non-tumor\n(combined)', 'Luminal\ndebris', 'Connective\ntissue', 'Necrosis', 'Liver\nparenchyma', 'Inflammation'], fontsize=11)
                # ax3.set_ylabel("Intensities [a.u.]")
                ax3.set_ylabel(r"$\rm norm. 2^{nd}~der.~of~trans.$")
                ax3.hlines(intercept, -0.5, 6.5, color="red", linestyle="--")
                for idx, (tissue, false_percentage_tissue) in enumerate(false_percentages_tissue.items()):
                    ax3.text(0.1 + 0.13 * idx, 0.02, f"{false_percentage_tissue:.0f}%", transform=ax3.transAxes, fontsize=11, color='red')
                ax3.text(0.05, 0.92, str(wavenumber_in_query) + r' $\rm cm^{-1}$', transform=ax3.transAxes, fontsize=13, color='green')
                ax3.margins(y=0.10)
                ax3.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
                # fig.savefig(os.path.join(self.save_imgs, 'error_test_wavenumber.pdf'), bbox_inches='tight')
                plt.show()

        # print(f"\npredictive_status:  {',  '.join(self.predictive_status_list)}")
        # self.feature_list_str = [" " + feature if len(feature) == 3 else feature for feature in self.feature_list_str]
        # print(f"feature_list:      {', '.join(self.feature_list_str)}")


    def calc_projection_image(self, feature_list: list[int] = [], predictive_status_list: list[str] = []) -> None:
        """
        Calculate the projection image from all wavenumbers either chosen by prior knowledge
        about the features or about the ROIs.

        Parameters:
            feature_list(list[int]): list of tissue type specific wavenumbers
            predictive_status_list(list[str]): list of predictive value as "pos" or "neg"
        """

        if feature_list != []:
            self.feature_list = feature_list
        if predictive_status_list != []:
            self.predictive_status_list = predictive_status_list

        print("Prior knowledge:", self.prior_knowledge)
        # Differentiate into two groups based on predictitve value
        positive_predictives: list[int] = []
        negative_predictives: list[int] = []
        for idx, feature in enumerate(self.feature_list):
            if self.predictive_status_list[idx] == "pos":
                positive_predictives.append(feature)
            elif self.predictive_status_list[idx] == "neg":
                negative_predictives.append(feature)
        print(f"PPFs: {positive_predictives} NPFs: {negative_predictives}")

        # Calculate Projection image for all samples
        for sample_number, _ in enumerate(self.filenames):

            # PPVs
            stackPPVIntensities = []
            for idx, positive_predictive in enumerate(positive_predictives):
                wave_num = positive_predictive
                data_stack_infiv_img = self.indiv_2darrays_preprocessed[sample_number]
                indexwavenumber2plot = np.where(self.second_deriv_wavenumbers == wave_num)[0]
                wave_num_img = data_stack_infiv_img[:, :, indexwavenumber2plot]
                stackPPVIntensities.append(wave_num_img[:,:,0])
            ppv_projection = np.zeros((len(wave_num_img), len(wave_num_img[0])))
            for idx, _ in enumerate(positive_predictives):
                ppv_projection = np.add(ppv_projection, stackPPVIntensities[idx])

            # NPVs
            stackNPVIntensities = []
            for idx, negative_predictive in enumerate(negative_predictives):
                wave_num = negative_predictive
                data_stack_infiv_img = self.indiv_2darrays_preprocessed[sample_number]
                indexwavenumber2plot = np.where(self.second_deriv_wavenumbers == wave_num)[0]
                wave_num_img = data_stack_infiv_img[:, :, indexwavenumber2plot]
                stackNPVIntensities.append(wave_num_img[:,:,0])
            npvProjection = np.zeros((len(wave_num_img), len(wave_num_img[0])))
            for idx, _ in enumerate(negative_predictives):
                npv_projection = np.add(npvProjection, stackNPVIntensities[idx])

            # Offest
            offset = 10
            # Projection image:
            projection = ppv_projection - npv_projection + offset
            self.projection_imgs.append(projection)

        fig, ax = plt.subplots(figsize=(10,10))
        print("Projection image:")
        im = ax.imshow(self.projection_imgs[0], cmap="magma")
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        cbar.ax.ticklabel_format(useOffset=False, style='plain')
        plt.show()


    def background_mask_calculation(self, mask_wavenumber: int) -> None:
        """
        Creation of binary mask (background=0, tissue=1)

        Parameters:
            mask_wavenumber(int): wavenumber used for masking (good results with raw data 1552 1/cm^(-1))
        """

        for filepath in self.file_paths:
            # Load IR raw data
            ir_spectra: Spectrum = specread(filepath)
            wavenumber = ir_spectra.wavelength #list of wavenumbers
            amplitudes_raw = ir_spectra.amplitudes
            meta = ir_spectra.meta
            image_dimensions = [meta["n_x"], meta["n_y"], meta["n_z"]]
            image_stack = amplitudes_raw.reshape(image_dimensions[1], image_dimensions[0], 
                                                image_dimensions[2])
            size_x = image_stack.shape[1]
            sizeY = image_stack.shape[0]
            index_individual_wavenumber = np.where(wavenumber == mask_wavenumber)
            int_pixels = amplitudes_raw[:, index_individual_wavenumber]
            int_img = int_pixels.reshape(sizeY, size_x)

            # Calculate individual masks for each section
            mask_data_flat = int_img.flatten()
            reshdata = mask_data_flat.reshape(-1,1)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(reshdata)
            kmeans.labels_
            binary = kmeans.labels_.reshape(len(int_img), len(int_img[0]))  # reshape in original size
            # Set pixels of tissue to be 1 and background to be 0
            if binary[0,0] == 1:
                mask = np.where(binary == 1, 0, 1)   # Invertiere binar 
            else:
                mask = binary 
            # Proccess mask
            mask_processed = ndimage.binary_closing(mask, structure=np.ones((7,7))).astype(int)
            self.indiv_maskarrays.append(mask_processed)

        # Show individual intermediate results
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(self.indiv_maskarrays[0])
        plt.show()


    def mask_projection_image(self):
        """
        Mask the individual projection image
        """
        # self.masked_project_imgs: list = []
        for projection_img, indiv_maskarray in zip(self.projection_imgs, self.indiv_maskarrays):
            # Mask background of projection image 
            # ... by exchanging intensitiy values with zero
            masked_project_img = indiv_maskarray * projection_img
            self.masked_project_imgs.append(masked_project_img)
            
        # For plotting results, the colorscale needs to be reset 
        # ... to min and max intensity values (!=0)
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(self.masked_project_imgs[0], cmap = "magma")
        im.set_clim(np.amin(self.masked_project_imgs[0][self.masked_project_imgs[0] != 0]),
                 np.amax(self.masked_project_imgs[0][self.masked_project_imgs[0] != 0]))
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        cbar.ax.ticklabel_format(useOffset=False, style='plain')
        plt.show()


    def save_masked_projection_images(self) -> None:
        """
        Save masked projection images for later significance analysis
        """

        parent_dir = os.path.basename(self.directory)
        save_subdir = 'masked_projection_imgs/'
        try:
            savedir_masked_projection = os.path.join(self.directory, save_subdir)
            os.makedirs(savedir_masked_projection, exist_ok=True)
            for idx, masked_project_img in enumerate(self.masked_project_imgs):
                file_id = re.split(r'[_.]', self.filenames[idx])[-2]
                np.save(os.path.join(savedir_masked_projection + f"maskedProjectionImg_{file_id}.npy"), masked_project_img)
            print(f"Masked projection images saved in '{parent_dir}/{save_subdir}'.")
        except AttributeError:
            raise AttributeError(f"Previous module(s) not executed to create data to save.")

