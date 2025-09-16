#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Nelson BC Serre, SICE team, RDP, CNRS, UMR5667, ENS de Lyon. 2025
Automatic segmentation and measurement of PIN2-GFP intracellular compartments.
"""

import tifffile
import pandas as pd
import numpy as np
import pyclesperanto_prototype as cle
from skimage.measure import regionprops_table
from scipy.ndimage import binary_fill_holes

# ---------------- USER INPUT ----------------
ecc_thresh_list = [0.7]
gammaVar_list = [1.8]
time = '15min-'
output_folder = "/Desired/Output/Folder/"

# ---------------- Helper classes & functions ----------------
class Experiment:
    def __init__(self, name, conditions, stack_path, stack_name):
        self.name = name
        self.conditions = conditions
        self.stack_path = stack_path
        self.stack_name = stack_name
        self.density_df = None
        self.area_df = None

    @property
    def full_stack_path(self):
        return self.stack_path + self.stack_name

    def set_density(self, df: pd.DataFrame):
        self.density_df = df

    def set_area(self, df: pd.DataFrame):
        self.area_df = df


def apply_mask(image, mask, label):
    """Apply a label mask to an image, keeping only pixels where mask == label."""
    mask_binary = (mask == label)
    if image.ndim == 2:
        return image * mask_binary
    elif image.ndim == 3:
        return image * mask_binary[:, :, np.newaxis]
    else:
        raise ValueError("Unsupported image shape")


def fill_mask_holes(labeled_mask):
    """Fill holes per labeled object while preserving labels."""
    filled_mask = np.zeros_like(labeled_mask)
    labels = np.unique(labeled_mask)
    labels = labels[labels != 0]  # exclude background
    for label in labels:
        obj_mask = (labeled_mask == label)
        filled_obj = binary_fill_holes(obj_mask)
        filled_mask[filled_obj] = label
    return filled_mask


# ---------------- Experiment definition (example) ----------------
exp1 = Experiment(
    name="Experiment-name",
    # Example: must match number of images in stack
    conditions=["Col-0_control"] * 43 + ["Col-0_ralf23"] * 73 +
               ["pip5ktriple_control"] * 19 + ["pip5ktriple_ralf23"] * 37,
    stack_path="/Experiment/Folder/Path/",
    stack_name="Stack.tif"
)

experiments = [exp1]


# ---------------- Main processing loop ----------------
combN = 1
total_combinations = len(ecc_thresh_list) * len(gammaVar_list)

for ecc_thresh in ecc_thresh_list:
    for gammaVar in gammaVar_list:
        savePath = f"{output_folder}/densityPlot_ec{ecc_thresh}_gamma{gammaVar}-"
        print(f'processing combination {combN}/{total_combinations}')

        for experiment in experiments:
            print(f"Processing experiment '{experiment.name}' with ecc_thresh={ecc_thresh}, gamma={gammaVar}")

            stack = tifffile.imread(experiment.full_stack_path)
            combined_df = None
            list_average_density = []

            # process each image in stack
            for image_n, image in enumerate(stack):
                print(f'  processing image {image_n+1}/{len(stack)}')

                # ---------- cell mask ----------
                remove_noise = cle.gaussian_blur(image, sigma_x=2, sigma_y=2, sigma_z=0)
                remove_bg = cle.subtract_gaussian_background(remove_noise, sigma_x=100, sigma_y=100, sigma_z=0)
                binary = cle.smaller_or_equal_constant(remove_bg, constant=1)
                binary_labelled = cle.connected_components_labeling_box(binary)
                binary_labelled_noedge = cle.exclude_labels_on_edges(binary_labelled)
                binary_labelled_noedge_big = cle.exclude_small_labels(binary_labelled_noedge, maximum_size=1000)
                binary_fill = fill_mask_holes(binary_labelled_noedge_big)
                binary_close = cle.closing_labels(binary_fill, radius=10)
                binary_erode = cle.erode_labels(binary_close, radius=15)
                binary_final = cle.pull(binary_erode)  # labeled mask

                # measure cell area
                table_label_area = regionprops_table(binary_final, properties=('label', 'area_filled'))
                table_label_area = pd.DataFrame(table_label_area)

                # ---------- per-cell loop ----------
                n_cell = int(np.max(binary_final))
                temp_list_density = []
                for maskid in range(1, n_cell + 1):
                    cell = apply_mask(remove_noise, binary_final, maskid)

                    row = table_label_area[table_label_area['label'] == maskid]
                    if row.empty:
                        continue
                    cell_area = float(row['area_filled'].values[0])

                    # detect intracellular objects
                    cell_blur = cle.gaussian_blur(cell, sigma_x=2, sigma_y=2, sigma_z=0)
                    cell_clean = cle.gamma_correction(cell_blur, gamma=gammaVar)
                    binary_cell = cle.threshold_otsu(cell_clean)
                    binary_label = cle.connected_components_labeling_box(binary_cell)
                    binary_clean = cle.exclude_labels_outside_size_range(binary_label, minimum_size=100, maximum_size=600)
                    binary_clean = cle.pull(binary_clean)

                    # measure eccentricity filter
                    label_table = regionprops_table(binary_clean, properties=('label', 'area_filled', 'eccentricity'))
                    label_df = pd.DataFrame(label_table)
                    label_df = label_df[label_df['eccentricity'] < ecc_thresh].reset_index(drop=True)

                    if label_df.shape[0] > 0:
                        label_df["image_n"] = image_n
                        label_df["maskid"] = maskid
                        if combined_df is None:
                            combined_df = label_df.copy()
                        else:
                            combined_df = pd.concat([combined_df, label_df], ignore_index=True)

                    # compute density (particles per 1000 pixel²)
                    cell_label_number = label_df.shape[0]
                    if cell_label_number > 0:
                        cell_label_density = (cell_label_number / cell_area) * 1000.0
                    else:
                        cell_label_density = 0.0
                    temp_list_density.append(cell_label_density)

                # per-image median density
                list_average_density.append(np.median(temp_list_density) if temp_list_density else 0.0)

            # ---------- save results ----------
            density_table = pd.DataFrame({
                'Image_index': np.arange(len(stack)),
                'Density endosome/1000 pixel2': list_average_density
            })
            if len(experiment.conditions) == len(stack):
                density_table["Condition"] = experiment.conditions
            else:
                density_table["Condition"] = ["unknown"] * len(stack)

            experiment.set_density(density_table)

            if combined_df is not None:
                condition_map = dict(zip(
                    range(0, len(stack)),
                    experiment.conditions if len(experiment.conditions) == len(stack) else ["unknown"] * len(stack)
                ))
                combined_df['condition'] = combined_df['image_n'].map(condition_map)
                experiment.set_area(combined_df)
                combined_df.to_csv(f"{savePath}{time}output_area.csv", index=False)
            else:
                experiment.set_area(pd.DataFrame())

            density_table.to_csv(f"{savePath}{time}output_density.csv", index=False)

        combN += 1

print("Processing completed. CSVs exported.")