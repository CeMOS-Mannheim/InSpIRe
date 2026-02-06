import os
import numpy as np # (1.22.3)
import matplotlib.pyplot as plt # (3.5.1)
from PIL import Image # (9.1.0)
from PIL.ImageFile import ImageFile
import csv # (1.0)
from matplotlib.path import Path # (3.5.1)
from skimage.color import rgb2gray # (0.19.2)
from scipy.ndimage import  rotate # (1.8.0)
import matplotlib.patches as mpatches # (3.5.1)
from matplotlib import colors

from ._coregistration_workflow import register_image_elastix, transformix_image

class BenchmarkingHotSpot:
    def __init__(self, dir_files: str) -> None:
        self.directory: str = dir_files
        self.benchmarking_example: str = os.path.join(dir_files, 'benchmarking_example_ID5a_Rec1/')
        self.hot_map: np.ndarray
        self.ir_coreg: np.ndarray
        self.image: np.ndarray
        self.zoom_factor: int
        self.corner_point_x: np.ndarray
        self.corner_point_y: np.ndarray
        self.width: np.ndarray
        self.height: np.ndarray
        self.zoom_in: np.ndarray
        self.tumor_annotationc_rec: np.ndarray
        self.angle: int
        self.mirror: bool
        self.grey: np.ndarray
        self.map_ir_2he: np.ndarray
        self.hot_map_transformed: np.ndarray
        self.zoom_in_hot_map: np.ndarray

        self.combined_to_compare: np.ndarray
        self.true_neg: int
        self.false_neg: int
        self.true_pos: int
        self.false_pos: int

        self.save_imgs = os.path.join(self.directory, os.pardir, 'imgs', 'add_on')
        # --- IR

        fig, (ax1, ax2) = plt.subplots(1, 2)
        # Hot spot map to be benchmarked
        hot_map_path = os.path.join(self.benchmarking_example, 'HotSpotMap_cohortwide_tumor_1proz_ID5a.npy')
        # self.hot_map = np.load(os.path.join(self.directory, "HotSpotMap_grouped_tumor_1proz_ID5a.npy"))
        self.hot_map = np.load(hot_map_path)
        cmap_high: colors.ListedColormap = colors.ListedColormap(['white', 'lightgrey', 'darkred'])
        boundaries: list[float] = [-15, -5, 0.5, 1.5]
        norm: colors.BoundaryNorm = colors.BoundaryNorm(boundaries, cmap_high.N, clip=True)
        ax1.set_title("Tumor hotspot map")
        ax1.imshow(self.hot_map, cmap=cmap_high, norm=norm)

        # IR raw data to calculate co-registration matrix
        ir_coreg_path = os.path.join(self.benchmarking_example, '1656_unprocessed_co-reg_ID5a.npy')
        # self.ir_coreg = np.load(os.path.join(self.directory, "ID5a_1656_co-reg_UNprocessed.npy"))
        self.ir_coreg = np.load(ir_coreg_path)
        ax2.set_title("MIR raw image")
        ax2.imshow(self.ir_coreg, cmap="magma")
        plt.show()


    def he_import(self) -> None:
        path_image = os.path.join(self.benchmarking_example, 'HE_40proz_ID5a.tif' )
        image: ImageFile = Image.open(path_image)
        self.image = np.array(image)
        fig, ax = plt.subplots()
        ax.imshow(self.image)
        plt.show()

    def he_zoomin(self, zoom_factor: int) -> None:
        # --- Pathology
        # Pathological annotation to benchmark against
        # Rectangle information within which precise annotations are provided
        self.zoom_factor = zoom_factor
        path_cleaned: str = os.path.join(self.benchmarking_example, 'annotation_Rec1_ID5a.csv')
        with open(path_cleaned)as csvdatei:
            csv_reader_rectangle: csv.DictReader[str] = csv.DictReader(csvdatei)
            corner_point_x: list[str] = []
            corner_point_y: list[str] = []
            width: list[str] = []
            height: list[str] = []

            for row in csv_reader_rectangle:
                types: str = row["type"]
                if types == "rectangle":
                    corner_point_x.append(row["X"])
                    corner_point_y.append(row["Y"])
                    width.append(row["Width"])
                    height.append(row["Height"])

        # self.corner_point_x = np.array(corner_point_x).astype("float64") / 10
        # self.corner_point_y = np.array(corner_point_y).astype("float64") / 10

        self.corner_point_x = np.array(corner_point_x).astype("float64") / 10
        self.corner_point_y = np.array(corner_point_y).astype("float64") / 10
        self.width = np.array(width).astype("float64") / 10
        self.height = np.array(height).astype("float64") / 10

        # Zoomin to annotated rectangle
        # zoom_factor = 4  # Data used for annotation was in 100% resolution
        # ... for data size reasons 40% resolution was used for co-registration and benchmarking
        self.zoom_in = self.image[int(self.corner_point_y[0] * zoom_factor):int(self.corner_point_y[0] * zoom_factor) +
                                  int(self.height[0] * zoom_factor), int(self.corner_point_x[0] * zoom_factor):
                                  int(self.corner_point_x[0] * zoom_factor) + int(self.width[0] * zoom_factor)]
        fig, ax = plt.subplots()
        ax.imshow(self.zoom_in)
        ax.axis("off")
        ax.grid(False)
        plt.show()


    def patho_annotation_import(self) -> None:
        # --- Pathology
        # Pathological annotation to benchmark against
        # Polygonal chains of all annotations within the regarding rectangle
        path_cleaned: str = os.path.join(self.benchmarking_example, 'annotation_Rec1_ID5a.csv')
        with open(path_cleaned) as csvdatei:
            csv_reader_object: csv.DictReader = csv.DictReader(csvdatei)
            polygons: list[str] = []
            tags: list[str] = []
            row: dict[str, str]
            for row in csv_reader_object:
                # print(row)
                polygons.append(row["Points"])
                tags.append(row["Text"])

        # The following step was repeated in the work for each tissue type
        # ... but here is only shown for tumorous tissue regions for simplicity
        tumor_polys: list[float] = []
        # for i in range(len(polygons)):
        for polygon, tag in zip(polygons, tags):
            indiv_polygon: str = polygon
            tag: str = tag

            if tag == "Tumor":
                integer: list[str] = indiv_polygon.split()  # 'x1,y1', 'x2,y2', ... individual x,y-coords as tuples
                x_coords: list[float] = []
                y_coords: list[float] = []
                # for m in range(len(integer)):  # for each individual tuple ...
                for coords in integer:
                    # coords = integer[m]
                    x, y = coords.split(",")  # split the touple in x and y coordinates
                    x = float(x) / 10  # Adapt to image resolution of download
                    y = float(y) / 10  # ""
                    x_coords.append(x)
                    y_coords.append(y)

                # Attach the first point again at the end to create a 'closed loop'
                x_coords.append(x_coords[0])
                y_coords.append(y_coords[0])
                tumor_polys.append((x_coords, y_coords))

        # Pathological annotation to benchmark against
        # Retrieve only the annotations of the currently tested tissue type (here tumor)
        # ... and prepare for comparison (transfer polygonal chain into pixel image)

        # Adapt for resolution factor of coordinate system in OMERO
        multiplicator = len(self.zoom_in) / self.width

        # TUMOR  (in original work repeated for all other tissue types as well)
        if 'Tumor' in tags:
            # Tumor polygon plot ----------------------------------------------------------
            x_coords_tumor: list[int] = []
            y_coords_tumor: list[int] = []
            for i in range(len(tumor_polys)):
                x_coords_individ = tumor_polys[i][0]
                x_coords_tumor.append(x_coords_individ)
                y_coords_individ = tumor_polys[i][1]
                y_coords_tumor.append(y_coords_individ)
            x_coords_tumor_rec = [(x - self.corner_point_x) * multiplicator for x in x_coords_tumor]
            y_coords_tumor_rec = [(y - self.corner_point_y) * multiplicator for y in y_coords_tumor]

            individ_binaries = []
            for i in range(len(x_coords_tumor_rec)):
                test_x = x_coords_tumor_rec[i]
                test_y = y_coords_tumor_rec[i]
                li = list(zip(test_x, test_y))
                nx, ny = len(self.zoom_in[0]), len(self.zoom_in)
                poly_verts = li
                x, y = np.meshgrid(np.arange(nx), np.arange(ny))
                x, y = x.flatten(), y.flatten()
                points = np.vstack((x, y)).T
                path = Path(poly_verts)
                grid = path.contains_points(points)
                grid = grid.reshape((ny, nx))
                individ_binaries.append(grid)
            binary_annotations = np.sum(individ_binaries, axis=0)
            self.tumor_annotationc_rec = np.where(binary_annotations > 0, 0.5, 0)

        else:
            print("No tumor regions annotated")
            self.tumor_annotationc_rec = np.zeros((len(self.zoom_in), len(self.zoom_in[0])))

        # Show results of annotation on original H&E stained section
        fig, ax = plt.subplots()
        ax.imshow(self.zoom_in)
        # Tumor
        ax.contour(self.tumor_annotationc_rec, colors="limegreen", linewidths=2)
        ax.contourf(self.tumor_annotationc_rec, colors="limegreen", alpha=0.5, levels=[0.25, 1.0])
        ax.grid(False)
        ax.axis("off")
        plt.show()


    def co_registration_matrix(self, angle: int, mirror: bool) -> None:
        # Co-register IR(moving) --> H&E(fixed)
        # Co-registration is performed between one dimensional images (e.g. greyscale)

        self.angle = angle
        self.mirror = mirror
        self.grey = rgb2gray(self.image)

        # Roughly align the infrared image with the H&E image
        # Original IR -----------------------------------------------------------------
        ir_coreg = self.ir_coreg
        # Mirror ----------------------------------------------------------------------
        # Flip YES
        if mirror:
            ir_coreg: np.ndarray = np.flip(ir_coreg, 1)
        # Flip NO
        #mirror_IRcoreg = IRcoreg
        # Rotate ----------------------------------------------------------------------

        ir_coreg = rotate(ir_coreg, angle)
        ir_coreg = ir_coreg[:,:,0]
        # H&E for comparison -----------------------------------------------------------

        # Calculate transformation matrix
        moved_ir_coreg, self.map_ir_2he = register_image_elastix(fixed = self.grey, moving = ir_coreg, 
                                                      method = 'bspline', iterations = 2000, 
                                                      fixed_pixel_size = 1.33/1000, 
                                                      moving_pixel_size = 25/1000)

        # Overlay results to visually check quality of alignment
        fig, ax = plt.subplots()
        ax.imshow(self.grey, cmap = "gist_gray")
        im = ax.imshow(moved_ir_coreg, cmap = "magma", alpha = 0.5)
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        cbar.ax.ticklabel_format(useOffset=False, style='plain')
        plt.show()

    def co_registration_transform(self) -> None:
        """

        """
        # Use co-registration map to transform hot spot map --> H&E
        # Slice map to the original size of according IR img
        sliced_hot_map: np.ndarray = self.hot_map[0:len(self.ir_coreg), 0:len(self.ir_coreg[0]),0]
        sliced_hot_map = np.where(sliced_hot_map == 1, 1, 0)


        # Orient both images the same way
        # Mirror ----------------------------------------------------------------------
        if self.mirror:
            sliced_hot_map = np.flip(sliced_hot_map, 1)
        # Rotate ----------------------------------------------------------------------
        sliced_hot_map = rotate(sliced_hot_map, self.angle)
        # Transform -------------------------------------------------------------------
        self.hot_map_transformed = transformix_image(image = sliced_hot_map, transform_parameter_map = self.map_ir_2he, 
                                     transform_pixel_size = 25/1000)
        self.hot_map_transformed = np.where(self.hot_map_transformed > 125, 1, 0)

        # Overlay results to visually check quality of alignment
        fig, ax = plt.subplots()
        ax.imshow(self.grey, cmap = "gist_gray")
        ax.imshow(self.hot_map_transformed, cmap = "Greens", alpha = 0.5)
        plt.show()

    def prob_map_zoomin(self) -> None:
        # Cut out the same rectangle from the hot spot map,
        # ... which has been annotated in the adjacent H&E section
        self.zoom_in_hot_map = self.hot_map_transformed[int(self.corner_point_y[0] * self.zoom_factor):
                                                        int(self.corner_point_y[0] * self.zoom_factor) +
                                                        int(self.height[0] * self.zoom_factor),
                                                        int(self.corner_point_x[0] * self.zoom_factor):
                                                        int(self.corner_point_x[0] * self.zoom_factor) +
                                                        int(self.width[0] * self.zoom_factor)]

        # Show results on original H&E stained section zoom in
        fig, ax = plt.subplots()
        ax.imshow(self.zoom_in)
        ax.contour(self.zoom_in_hot_map, colors="limegreen", linewidths=2)
        ax.contourf(self.zoom_in_hot_map, colors="limegreen", alpha=0.5, levels=[0.25, 1.0])
        ax.grid(False)
        ax.axis("off")
        plt.show()

    def compare_patho_prob_map(self) -> None:
        # Calculate confusion matrix

        # Both annotations are combined into one image to directly compare the results
        self.combined_to_compare = self.zoom_in_hot_map + self.tumor_annotationc_rec
        # True negative (true_pos), false negative (false_neg), false positive (self.false_pos),
        # and true positive (self.true_pos)
        self.true_neg = int(np.count_nonzero(self.combined_to_compare == 0))
        self.false_neg = int(np.count_nonzero(self.combined_to_compare == 0.5))
        self.false_pos = int(np.count_nonzero(self.combined_to_compare == 1))
        self.true_pos = int(np.count_nonzero(self.combined_to_compare == 1.5))
        # totalCount = self.true_neg + self.false_neg + self.false_pos + self.true_pos


    def benchmarking(self) -> None:
        """

        """
        accuracy = (self.true_pos + self.true_neg) / (self.true_pos + self.true_neg +
                                                      self.false_pos + self.false_neg) * 100
        print("Accuracy:", np.round(accuracy, 2), "%")
        # Plot results
        # Define suitable colormap
        cmap_acc = colors.ListedColormap(['lightgreen', 'red', 'darkred', 'green'])
        boundaries = [-0.5, 0.25, 0.75, 1.25, 1.75]
        norm_acc = colors.BoundaryNorm(boundaries, cmap_acc.N, clip=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.7, hspace=0.7)
        ax1.set_title("Accuracy")
        ax1.imshow(self.combined_to_compare, cmap=cmap_acc, norm=norm_acc)
        ax1.grid(False)
        ax1.axis("off")
        true_neg_patch = mpatches.Patch(color='lightgreen', label='True negative')
        false_neg_patch = mpatches.Patch(color='red', label='False negative')
        false_pos_patch = mpatches.Patch(color='darkred', label='False positive')
        true_pos_patch = mpatches.Patch(color='green', label='True positive')
        ax1.legend(handles=[true_neg_patch, false_neg_patch, false_pos_patch, true_pos_patch],
                   loc='center left', bbox_to_anchor=(1, 0.5))

        precision = (self.true_pos) / (self.true_pos + self.false_pos) * 100
        print("Precision:", np.round(precision, 2), "%")
        # Plot results
        # Define suitable colormap
        cmap_precision = colors.ListedColormap(['lightgray', 'lightgray', 'darkred', 'green'])
        boundaries = [-0.5, 0.25, 0.75, 1.25, 1.75]
        norm_precision = colors.BoundaryNorm(boundaries, cmap_precision.N, clip=True)
        ax2.set_title("Precision")
        ax2.imshow(self.combined_to_compare, cmap=cmap_precision, norm=norm_precision)
        ax2.grid(False)
        ax2.axis("off")
        false_pos_patch = mpatches.Patch(color='darkred', label='False positive')
        true_pos_patch = mpatches.Patch(color='green', label='True positive')
        not_considered = mpatches.Patch(color='lightgray', label='Not considered')
        ax2.legend(handles=[false_pos_patch, true_pos_patch, not_considered], loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

