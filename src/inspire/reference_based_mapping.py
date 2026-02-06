import os
import re
from glob import glob
import pandas as pd # (1.4.2)
import numpy as np  # (1.22.3)
np.random.seed(420)
import matplotlib.pyplot as plt   # (3.5.1)
from matplotlib import colors
import math
import libpysal as lps # (4.6.2)
import geopandas as gpd # (0.9.0)
from esda import Moran_Local # (2.4.1)

# Introduce suiting color map
cmapHigh = colors.ListedColormap(['white', 'lightgrey', 'darkred'])
boundaries = [-15, -5, 0.5, 1.5]
norm = colors.BoundaryNorm(boundaries, cmapHigh.N, clip=True)


def raster_2darrays(list_imgs, wavelength2plot=0, data_wavelengths=0, with_mask=True):
    w_max = np.amax([x['image'].shape[0] for x in list_imgs])
    h_max = np.amax([x['image'].shape[1] for x in list_imgs])
    super_img = dict(
        hw_max=[w_max, h_max],
        original_hw={})
    num_col = len(list_imgs)
    num_empty = math.ceil(len(list_imgs) / num_col) * num_col - len(list_imgs)
    images, row = [], []
    masks, row_mask = [], []
    row_counter = 0
    file_name = []
    for i in range(len(list_imgs) + num_empty):
        if i < len(list_imgs):
            dataset = list_imgs[i]['image']
            img = np.zeros((w_max, h_max))
            img[:dataset.shape[0], :dataset.shape[1]] = dataset

            row.append(img)
            super_img['original_hw'][str(row_counter) + '_' + str(i - (num_col * row_counter))] \
                = img.shape[:2]
            if with_mask:
                dataset = list_imgs[i]['mask']
                mask = np.zeros((w_max, h_max))
                mask[:dataset.shape[0], :dataset.shape[1]] = dataset
                row_mask.append(mask)
        else:
            row.append(np.zeros((w_max, h_max)))
            if with_mask:
                row_mask.append(np.zeros((w_max, h_max)))
        if len(row) == num_col:
            images.append(np.hstack(row))
            row = []
            if with_mask:
                masks.append(np.hstack(row_mask))
                row_mask = []
            row_counter += 1
    super_img['global_img'] = np.vstack(images)
    super_img['mask'] = np.vstack(masks)
    return super_img


def plot_img(img):
    plt.imshow(img, cmap = "magma")
    plt.clim(np.amin(img[img != 0]),np.amax(img[img != 0]))
    plt.axis('off')
    plt.colorbar(fraction=0.01, pad=0.04)
    plt.show()


def calc_hotspot(global_image, threshold=0.01, n_jobs=-1, save_path=None):
    mask = global_image['mask']
    input_img = global_image['global_img'] * mask
    height, width = input_img.shape
    df_database = create_dataframe(input_img, mask)
    # Calculate Spatial attribute
    wqnew = lps.weights.Queen.from_dataframe(df_database)
    wqnew.transform = 'r'
    ynew = df_database['intensities']
    ylagnew = lps.weights.lag_spatial(wqnew, ynew)
    linew = Moran_Local(ynew, wqnew, n_jobs=n_jobs)
    # Extract only the probabilistic values
    signew = linew.p_sim < threshold
    highspotnew = signew * linew.q == 1
    lowspotnew = signew * linew.q == 3
    hotspotmap = create_map(highspotnew, mask)
    lowspotmap = create_map(lowspotnew, mask)
    original_hw = [o for o in global_image['original_hw'].values()]

    linew_dict = dict(hotspotmap=hotspotmap,
                      lowspotmap=lowspotmap,
                      mask=global_image['mask'],
                      global_img=global_image['global_img'],
                      p_sim=linew.p_sim,
                      q=linew.q,
                      hw_max=global_image['hw_max'],
                      original_hw=original_hw,
                      linew=linew)
    return linew_dict


def create_dataframe(input_img, mask):
    height, width = input_img.shape
    points, intensity = [], []
    for h in range(height):
        for w in range(width):
            if mask[h,w] > 0:
                points.append(f'POINT({w} {h})')
                intensity.append(input_img[h,w])
    # Create data frame out of data
    frame = {'intensities': intensity, 'wkt': points}
    dataframe = pd.DataFrame(frame)
    geoseries = gpd.GeoSeries.from_wkt(dataframe['wkt'])
    geodataframe = gpd.GeoDataFrame(dataframe, geometry=geoseries, crs="EPSG:4326")
    return geodataframe


def create_map(spots, mask):
    copy = mask.flatten()
    bgcopy = np.where(copy == 0, -10, copy)
    indxa = np.where(bgcopy == 1)
    bgcopy[indxa] = spots
    return bgcopy.reshape(mask.shape)


def cal(ref_file, test_file, linew, idx):
    max_miou = 0
    h_max, w_max = test_file['hw_max']
    for thresh in range(1, 100, 1):
        thresh *= 0.005
        signew = linew.p_sim < thresh
        highspotnew = signew * linew.q==1
        highplot = create_map(highspotnew, test_file['mask'])
        _h_min, _w_min = 0, idx * w_max
        _h_max = test_file['original_hw'][idx][0]
        _w_max = idx * w_max + test_file['original_hw'][idx][1]
        hotspot = highplot[_h_min:_h_max,_w_min:_w_max]
        mse = ((ref_file-hotspot)**2).mean()
        miou, class_iou = compute_iou(hotspot, ref_file)
        if miou > max_miou:
            max_miou = miou
            min_mse = mse
            best_thresh = thresh
            best_class_iou = class_iou['1']
            best_highplot = hotspot

    print(f'### Best results @ thresh = {best_thresh} ###')
    print(f'    Mean Squared Error {min_mse:.4f}')
    print(f'    Mean IoU {max_miou:.4f}')
    print(f'    HotSpot IoU {best_class_iou:.4f}')

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10, 5))
    ax1.imshow(best_highplot, cmap = cmapHigh, norm = norm)
    ax1.axis("off")   # turns off axes
    ax1.axis("tight")  # gets rid of white border
    ax1.axis("image")  # square up the image instead of filling the "figure" space
    ax1.title.set_text('Hotspot map database')

    ax2.imshow(ref_file, cmap = cmapHigh, norm = norm)
    ax2.axis("off")   # turns off axes
    ax2.axis("tight")  # gets rid of white border
    ax2.axis("image")  # square up the image instead of filling the "figure" space
    ax2.title.set_text('Hotspot map reference sample')
    plt.show()
    return best_thresh


def compute_iou(y_pred, y_true):
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    miou = []
    class_iou = {}
    for c in np.unique(y_true):
        tp = np.count_nonzero((y_true==c) * (y_pred==c))
        fp = np.count_nonzero((y_true!=c) * (y_pred==c))
        fn = np.count_nonzero((y_true==c) * (y_pred!=c))
        iou = tp / (tp + fp + fn)
        class_iou[str(int(c))] = iou
        miou.append(iou)
    return sum(miou)/len(miou), class_iou


class ReferenceBasedMapping:
    def __init__(self, directory: str) -> None:

        self.directory: str = directory
        self.masked_projection: str = os.path.join(self.directory, 'masked_projection_imgs')
        self.probability_maps: str = os.path.join(self.directory, 'probability_maps')
        self.pre_proc_database: list[np.ndarray] = []
        self.hot_spot_database: list[np.ndarray] = []
        self.pre_proc_new_sample: np.ndarray
        self.low_ref_hot_spot: np.ndarray 
        self.low_ref_pre_proc: np.ndarray
        self.high_ref_hot_spot: np.ndarray
        self.high_ref_pre_proc: np.ndarray
        self.path_pre_proc_new_sample: str
        # self.global_test
        # self.test_map
        # self.map_x

        self.save_imgs: str = os.path.join(self.directory, 'imgs', 'rbm')

        # Pre-processed data
        filepaths_prepro: list[str] = sorted(glob(os.path.join(self.masked_projection + '/*')), key=os.path.basename)

        path: str
        for path in filepaths_prepro:
            if path.endswith(".npy"):
                pre_proc_data: np.ndarray = np.load(path)
                self.pre_proc_database.append(pre_proc_data)

        # Hot spot maps
        directory_hotspot = os.path.join(self.probability_maps, 'cohort_wide_mapping')
        filepaths_hotspot = sorted(glob(os.path.join(directory_hotspot + '/*')), key=os.path.basename)
        for path in filepaths_hotspot:
            if path.endswith(".npy"):
                hot_spot_data: np.ndarray = np.load(path)
                self.hot_spot_database.append(hot_spot_data)

        # Newly added sample (pre-processed) -------------------------------------
        self.path_pre_proc_new_sample = os.path.join(self.probability_maps, 'reference_based_mapping', "maskedProjectionImg_ID25.npy")
        self.pre_proc_new_sample = np.load(self.path_pre_proc_new_sample)

        fig, axs = plt.subplots(1,len(self.hot_spot_database),figsize=(4*len(self.hot_spot_database),4))
        for idx, hotSpotData in enumerate(self.hot_spot_database):
            #TODO: Was 'Tissue' before
            # axs[idx].set_title(f"Tissue: {idx}")
            axs[idx].set_title(f"database sample: {idx+1}")
            axs[idx].imshow(hotSpotData, cmap = cmapHigh, norm = norm)
        plt.show()

    def select_reference_files(self) -> None:
        # Select the reference files from the database
        # ... from the lowest and the highest hot spot to total area ratio
        hotspot2totalAreaRatio = []
        for i in range(len(self.hot_spot_database)):
            sizeHotspot = np.count_nonzero(self.hot_spot_database[i] == 1)
            sizeTotal = np.count_nonzero(self.hot_spot_database[i] == 0) + sizeHotspot
            ratio = sizeHotspot / sizeTotal * 100
            hotspot2totalAreaRatio.append(ratio)
        # Find the lowest
        idx = int((np.where(hotspot2totalAreaRatio == np.amin(hotspot2totalAreaRatio)))[0][0])
        self.low_ref_hot_spot = self.hot_spot_database[idx]
        self.low_ref_pre_proc = self.pre_proc_database[idx]
        lowest_ratio = hotspot2totalAreaRatio[idx]
        # Find the highest
        idx = int((np.where(hotspot2totalAreaRatio == np.amax(hotspot2totalAreaRatio)))[0][0])
        self.high_ref_hot_spot = self.hot_spot_database[idx]
        self.high_ref_pre_proc = self.pre_proc_database[idx]
        highest_ratio = hotspot2totalAreaRatio[idx]

        print("Hotspot-to-total tissue area ratio for:")
        print(f"Low reference: {lowest_ratio:.1f} %")
        print(f"High reference: {highest_ratio:.1f} %")

        # Plot references used for pseudo-grouped probabilistic mapping
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,20))
        ax1.imshow(self.low_ref_hot_spot, cmap=cmapHigh, norm=norm)
        ax1.title.set_text('Low reference')
        ax2.imshow(self.high_ref_hot_spot, cmap=cmapHigh, norm=norm)
        ax2.title.set_text('High reference')
        plt.show()

    def combine_references_target(self) -> None: # was combine_reference_target
        list_dicts = [
            dict(image=self.low_ref_pre_proc, mask=np.where(self.low_ref_pre_proc == 0, 0, 1)),
            dict(image=self.high_ref_pre_proc, mask=np.where(self.high_ref_pre_proc == 0, 0, 1)),
            dict(image=self.pre_proc_new_sample, mask=np.where(self.pre_proc_new_sample == 0, 0, 1))]

        # Globalize
        self.global_test = raster_2darrays(
            list_imgs=list_dicts,
            wavelength2plot=0,
            data_wavelengths=0,
            with_mask=True)

        plot_img(self.global_test['global_img'])

    def calc_reference_based_map(self) -> None:
        path_test = self.directory + "C4_Pseudo-grouped significance mapping/"
        linew_test = calc_hotspot(self.global_test, save_path=path_test)
        self.test_map = linew_test
        test_h_max, test_w_max = self.test_map['hw_max']

        # Obtain resulting hot spot maps
        file_test = self.test_map['hotspotmap'][:, 0:test_w_max]
        file_test = file_test[:self.test_map['original_hw'][0][0], :self.test_map['original_hw'][0][1]]

        ref1_test = self.test_map['hotspotmap'][:, test_w_max:2 * test_w_max]
        ref1_test = ref1_test[:self.test_map['original_hw'][1][0], :self.test_map['original_hw'][1][1]]

        ref2_test = self.test_map['hotspotmap'][:, 2 * test_w_max:]
        ref2_test = ref2_test[:self.test_map['original_hw'][2][0], :self.test_map['original_hw'][2][1]]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        for idx in range(3):
            ax1.imshow(file_test, cmap=cmapHigh, norm=norm)
            ax1.axis("off")  # turns off axes
            ax1.axis("tight")  # gets rid of white border
            ax1.axis("image")  # square up the image instead of filling the "figure" space
            ax2.imshow(ref1_test, cmap=cmapHigh, norm=norm)
            ax2.axis("off")  # turns off axes
            ax2.axis("tight")  # gets rid of white border
            ax2.axis("image")  # square up the image instead of filling the "figure" space
            ax3.imshow(ref2_test, cmap=cmapHigh, norm=norm)
            ax3.axis("off")  # turns off axes
            ax3.axis("tight")  # gets rid of white border
            ax3.axis("image")  # square up the image instead of filling the "figure" space
        plt.show()

    def confidence_level_adjustment(self) -> None: #was adjust confidence level
        # Database hotspot maps for cutoff-value adaptation

        # Ref1
        ref1_database = self.low_ref_hot_spot[:self.test_map['hw_max'][0], :self.test_map['hw_max'][1], 0]
        # Ref2
        ref2_database = self.high_ref_hot_spot[:self.test_map['hw_max'][0], :self.test_map['hw_max'][1], 0]

        # Calculate optimum of cutoff-value
        # ... for lower reference
        self.thresh_1 = cal(ref1_database, self.test_map, self.test_map['linew'], 0)

        # Calculate optimum of cutoff-value
        # ... for upper reference
        self.thresh_2 = cal(ref2_database, self.test_map, self.test_map['linew'], 1)

    def probabilistic_added_sample(self) -> None:
        # Use the adapted cutoff-value to calculate the probabilistic map of the newly added sample
        # ...  in coherence to the original cohort
        w_max = self.test_map['hw_max'][1]

        thresh = (self.thresh_1 + self.thresh_2) / 2
        signew = self.test_map['linew'].p_sim < thresh
        highspotnew = signew * self.test_map['linew'].q == 1
        highplot = create_map(highspotnew, self.test_map['mask'])

        _h_min, _w_min = 0, 2 * w_max
        _h_max, _w_max = self.test_map['original_hw'][0][0], self.test_map['original_hw'][0][1]
        self.map_x = highplot[_h_min:_h_max, _w_min:(_w_min + _w_max)]
        # Plot result
        plt.imshow(self.map_x, cmap=cmapHigh, norm=norm)
        plt.axis("off")  # turns off axes
        plt.axis("tight")  # gets rid of white border
        plt.axis("image")  # square up the image instead of filling the "figure" space
        plt.show()
    
    def save_probabilistic_map(self) -> None:
        """

        """
        save_subdir = os.path.join(self.probability_maps, 'reference_based_mapping')
        parent_dir = self.directory.split('/')[-1]
        file_id = re.split(r'[_.]', self.path_pre_proc_new_sample)[-2]
        savedir = os.path.join(self.directory, save_subdir, f"HotSpotMap_reference_based_tumor_{file_id}.npy")
        np.save(savedir, self.map_x)
        print(f"Map succesfully saved to '{parent_dir}/{save_subdir}'.")

                # self.pre_proc_new_sample = np.load(self.directory
                #                    + "C4_Pseudo-grouped significance mapping/Newly added sample/"
                #                    + "maskedProjectionImg_ID25.npy")
