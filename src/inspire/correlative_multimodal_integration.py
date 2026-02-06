import os
from typing import Any, Literal

from matplotlib import colors # (3.10.7)
from matplotlib import pyplot as plt
import numpy as np  # (2.2.6)
import numpy.typing as npt 
from PIL import Image  # (12.0.0)
from scipy.ndimage import  rotate  # (1.16.2)
import seaborn as sns  # (0.13.2)
import SimpleITK as sitk  # (SimpleITK-SimpleElastix 2.5.0.dev49)
from skimage.transform import resize_local_mean  # (0.25.2)

from ._coregistration_workflow import register_image_elastix, transformix_image


class CorrelativeMultimodalIntegration:
    def __init__ (self, data_directory: str):
        """
        Do some whatever.

        Parameter:
            data_directory: location of data for loading and saving
            datatype: equidistant (staining data) or irregular (ir images)
        """

        self.directory: str = data_directory
        self.multimodal_correlation: str = os.path.join(self.directory, 'multimodal_correlation')
        self.hoechst: npt.NDArray[np.uint8]
        self.cold_map_binary: npt.NDArray[np.uint8]
        self.ir_hotspot_map: npt.NDArray[np.int32]
        self.cmap_high: colors.Colormap
        self.norm: colors.BoundaryNorm

        self.origin_x: int
        self.origin_y: int
        self.size_x: int
        self.size_y: int
        self.teach: npt.NDArray[np.int8]
        self.sliced_teach: npt.NDArray[np.int8]
        self.mirror: bool
        self.angle: int
        self.maphoechst: sitk.ParameterMap
        self.ir_raw_coreg: npt.NDArray[np.float32]
        self.map_ir_raw_coreg: Any
        self.transformed_ir_hotspotmap_ternary: Any
        self.teaching_binary_cold_map: Any
        # self.teaching_binary_cold_map: np.ma.MaskedArray[bool, bool]
        self.teaching_binary_hot_map: Any

        # Load the significance map results from IR imaging
        self.ir_hotspot_map = np.load(os.path.join(self.multimodal_correlation, 'HotSpotMap_cohortwide_tumor_1proz_ID32.npy'))
        self.cmap_high = colors.ListedColormap(['white', 'lightgrey', 'darkred'])
        boundaries = [-15, -5, 0.5, 1.5]
        self.norm = colors.BoundaryNorm(boundaries, self.cmap_high.N, clip=True)
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(len(self.ir_hotspot_map[0])/50, len(self.ir_hotspot_map)/100))
        ax1.imshow(self.ir_hotspot_map, cmap=self.cmap_high, norm=self.norm)
        hot_map_binary = np.where(self.ir_hotspot_map == 1, 1, 0)
        ax2.imshow(hot_map_binary)
        plt.show()

    def slice_sample_teaching(self, origin_x: int, origin_y: int, size_x: int, size_y: int) -> None:
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.size_x = size_x
        self.size_y = size_y
        # Load teaching image of MSI acquisition ---------------------------------

        path_teaching_iri = os.path.join(self.multimodal_correlation, 'teaching_ID32-9-30_MIRMSI.tif')
        teach = Image.open(path_teaching_iri).convert('L')
        self.teach = np.array(teach)
        _, ax = plt.subplots()
        ax.imshow(teach, cmap='gist_gray')
        plt.show()
        # Cut out the relevant section for co-registration -----------------------
        self.sliced_teach = self.teach[self.origin_y:self.origin_y+self.size_y,
                           self.origin_x:self.origin_x+self.size_x]
        _, ax = plt.subplots()
        ax.imshow(self.sliced_teach, cmap='gist_gray')
        plt.show()

    def co_registration_matrix(self, mirror: bool, angle: int) -> None:
        self.mirror = mirror
        self.angle = angle

        # Morphological detail in both images are required for co-registration ---
        # ... thus the IR image is more suitable to be used (ir_raw, 1656 cm^(-1))
        path_ir_raw_coreg = os.path.join(self.multimodal_correlation, '1656_unprocessed_co-reg_ID32.npy' )
        self.ir_raw_coreg = np.load(path_ir_raw_coreg)
        # Co-register reference image (ir_raw) with teaching image (MSI)
        # ... first roughly align
        # Original IRrawCoreg
        # Mirror
        if self.mirror:
            self.ir_raw_coreg = np.flip(self.ir_raw_coreg, 1)
        # Rotate
        self.ir_raw_coreg = rotate(self.ir_raw_coreg, self.angle)
        # Resize
        self.ir_raw_coreg = self.ir_raw_coreg[:, :, 0]
        # Teaching image for comparison 
        # ... then calculate co-registration matrix
        moved_ir_raw_coreg, self.map_ir_raw_coreg = register_image_elastix(fixed = self.sliced_teach,
                                moving=self.ir_raw_coreg, method='bspline', iterations=2000,
                                fixed_pixel_size=5/1000, moving_pixel_size=25/1000)
        _, ax = plt.subplots()
        ax.imshow(self.sliced_teach, cmap='gist_gray')
        ax.imshow(moved_ir_raw_coreg, alpha = 0.5, cmap='magma')
        plt.show()

    def co_registration_transform(self) -> None:

        # Cut the hot spot map back to the same size of the original IR image
        ir_hotspotmap_cut: npt.NDArray[np.int32] = self.ir_hotspot_map[0:len(self.ir_raw_coreg), 0:len(self.ir_raw_coreg[0])]
        # Increase the difference between hot spot region (=1) and non-sig. region (=0)
        ir_hotspotmap_cut = np.where(ir_hotspotmap_cut == 1, 10, ir_hotspotmap_cut)
        # ... first roughly align
        # Mirror
        if self.mirror:
            ir_hotspotmap_cut = np.flip(ir_hotspotmap_cut, 1)
        # Rotate
        ir_hotspotmap_cut = rotate(ir_hotspotmap_cut, self.angle)
        # Resize
        ir_hotspotmap_cut = ir_hotspotmap_cut[..., 0]
        # ... then use the co-registration matrix
        transformed_ir_hotspotmap = transformix_image(image=ir_hotspotmap_cut,
                                                      transform_parameter_map=self.map_ir_raw_coreg,
                                                      transform_pixel_size=25/1000)
        transformed_ir_hotspotmap_stepone = np.where(transformed_ir_hotspotmap < 75, 0, transformed_ir_hotspotmap)
        transformed_ir_hotspotmap_steptwo = np.where(transformed_ir_hotspotmap_stepone > 180, 1,
                                                     transformed_ir_hotspotmap_stepone)
        self.transformed_ir_hotspotmap_ternary = np.where(transformed_ir_hotspotmap_steptwo > 1, 0,
                                                          transformed_ir_hotspotmap_steptwo)
        
        self.transformed_ir_hotspotmap_ternary = np.ma.masked_where(self.transformed_ir_hotspotmap_ternary  == 0, self.transformed_ir_hotspotmap_ternary)
        _, ax = plt.subplots()
        ax.imshow(self.sliced_teach, cmap='gist_gray')
        ax.imshow(self.transformed_ir_hotspotmap_ternary, cmap=self.cmap_high, norm=self.norm)
        plt.show()

    def fit_referenced_teaching(self) -> None:
        # Resize to MSI teaching file
        # Transfer ternary to binary
        transformed_ir_hotspotmap_binary = np.where(self.transformed_ir_hotspotmap_ternary == 1, 1, 0)
        # Transfer the ROIs back on the original teaching tif file
        teaching_binary_hot_map = np.zeros((len(self.teach), len(self.teach[0])))
        teaching_binary_hot_map[self.origin_y:self.origin_y+self.size_y, self.origin_x:self.origin_x+self.size_x] = \
                                transformed_ir_hotspotmap_binary
        # transfer to data type bool to require less space
        teaching_binary_hot_map = teaching_binary_hot_map.astype('bool')

        self.teaching_binary_hot_map = np.ma.masked_where(teaching_binary_hot_map == 0, teaching_binary_hot_map)
        # Check before export
        _, ax = plt.subplots()
        ax.imshow(self.teach, cmap='gist_gray')
        ax.imshow(self.teaching_binary_hot_map, cmap='Reds', vmin=0, vmax=1.2)
        plt.show()

    def export_reference_teaching(self) -> None:

        # Export 
        im = Image.fromarray(self.teaching_binary_hot_map)
        save_path_hotmap = os.path.join(self.multimodal_correlation, 'HotSpotMap_sizeteachingimg_ID32.tif')
        im.save(save_path_hotmap)
        print("HotSpotMap_sizeteachingimg saved in multimodal_correlation/")

