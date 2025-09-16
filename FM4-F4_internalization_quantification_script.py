#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Nelson BC Serre, SICE team, RDP, CNRS, UMR5667, ENS de Lyon. 2025
Fm4-64 cell segmentation to calculate area and then measure spots
"""


from skimage.io import imread
import pyclesperanto_prototype as cle  # version 0.24.2
from skimage.segmentation import relabel_sequential as sk_relabel_sequential
from skimage.segmentation import clear_border
import numpy as np
from skimage import measure, exposure
import imageio
import os
import pandas as pd
import warnings
from datetime import date

output_folder = "output-folder"

date = date.today()


def remove_labels_on_edges(label_image):
    result = clear_border(np.asarray(label_image))
    relabeled_result, _, _ = sk_relabel_sequential(result)
    return relabeled_result

def segment_individual_cells(image):
    image1_D = cle.difference_of_gaussian(image, None, 100.0, 100.0, 0.0, 2.0, 2.0, 0.0)
    image2_gc = cle.greater_constant(image1_D, None, 1.0)
    image3_cclb = cle.connected_components_labeling_box(image2_gc)
    image4_R = remove_labels_on_edges(image3_cclb)
    image5_esl = cle.exclude_small_labels(image4_R, None, 10000.0)
    image6_sl = cle.smooth_labels(image5_esl, None, 5.0)
    image7_elwmr = cle.dilate_labels(image6_sl, None, 20.0)
    image8_el = cle.erode_labels(image7_elwmr, None, 25, False)
    image8_el = (image8_el * 255).astype(np.uint8) if image8_el.dtype == np.float64 else image8_el.astype(np.uint8)
    image9 = cle.mode_box(image8_el, None, 25.0, 25.0, 0.0)
    return cle.pull(image9)

def crop_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return mask[rmin:rmax+1, cmin:cmax+1]

def circularity_filter(segmentation_image, threshold):
    filtered_mask = np.zeros_like(segmentation_image)
    properties = measure.regionprops(segmentation_image)
    for prop in properties:
        area = prop.area
        perimeter = prop.perimeter
        if perimeter == 0:
            circularity = 0
        else:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity >= threshold:
            filtered_mask[segmentation_image == prop.label] = prop.label
    return filtered_mask

# ---------------- USER DATA ----------------
data = {
    "Col0_Control": 'folder_path',
    "Col0_RALF23": 'folder_path',
    "pip5k7,8,9_Control": 'folder_path',
    "pip5k7,8,9_RALF23": 'folder_path'
}

# ---------------- MAIN LOOP ----------------
imagesnumber = sum(len([f for f in os.listdir(path) if f.endswith(".tif")]) for path in data.values())
image_counter = 0
list_conditions = list(data.keys())

for condition, images_path in data.items():
    celln = 0
    image_list = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(".tif")]
    try:
        os.mkdir(f'{images_path}/segmentations')
    except FileExistsError:
        pass

    for image in image_list:
        image_counter += 1
        print(f'Image {image_counter}/{imagesnumber}')
        image0 = imread(image)
        image0_name = os.path.basename(image)
        cells_segmentation = segment_individual_cells(image0)
        imageio.imwrite(f'{images_path}/segmentations/{image0_name}_segmented.tif', cells_segmentation, format='tiff')

        scale = 0.0517902
        n_cell = int(np.max(cells_segmentation))
        cell_area_list, cell_density_list, particle_area_list = [], [], []
        cell_filename_list, cell_condition_list = [], []

        for i in range(1, n_cell + 1):
            print(f'Condition: {condition} ----- cell: {celln}')
            indiv_cell_segmentation = cells_segmentation.copy() == i
            indiv_cell_signal0 = image0.copy() * indiv_cell_segmentation
            indiv_cell_signal1 = crop_mask(indiv_cell_signal0)
            imageio.imwrite(f'{images_path}/segmentations/{image0_name}_cell{i}_original.tif',
                            indiv_cell_signal1, format='tiff')

            labeled_image, _ = measure.label(indiv_cell_segmentation, return_num=True)
            cell_properties = measure.regionprops(labeled_image)
            if not cell_properties:
                continue
            cell_area = cell_properties[0].area * scale**2
            cell_area_list.append(cell_area)
            cell_filename_list.append(image0_name)
            cell_condition_list.append(condition)

            # Measure spots
            image1 = exposure.equalize_adapthist(indiv_cell_signal1, clip_limit=0.005)
            image1_thb = cle.top_hat_box(image1, None, 10.0, 10.0, 0.0)
            image2_G = cle.gauss_otsu_labeling(image1_thb, None, 2.0)
            image3_ecl = cle.erode_connected_labels(image2_G, None, 3.0)
            image5_rs = cle.relabel_sequential(image3_ecl, None, 2.0)
            labeled_image, num_particles = measure.label(image5_rs, return_num=True)
            labeled_image = circularity_filter(labeled_image, 0.3)
            imageio.imwrite(f'{images_path}/segmentations/{image0_name}_cell{i}_label.tif',
                            labeled_image, format='tiff')

            warnings.filterwarnings("ignore", message=".*Low image data range.*")
            if num_particles / cell_area <= 1:
                cell_density_list.append(num_particles / cell_area)
            else:
                cell_density_list.append(np.nan)

            for particle in measure.regionprops(labeled_image):
                particle_area_list.append(particle.area * scale**2)

            listname = [image0_name] * len(particle_area_list)
            listcondition = [condition] * len(particle_area_list)
            listcellnumber = [i] * len(particle_area_list)

            particle_df_temp = pd.DataFrame({
                "Filename": listname,
                "Condition": listcondition,
                "Cell": listcellnumber,
                "Compartement_area": particle_area_list
            })

            if 'particle_df' not in locals():
                particle_df = particle_df_temp
            else:
                particle_df = pd.concat([particle_df, particle_df_temp], ignore_index=True)

            celln += 1

        cell_df_temp = pd.DataFrame({
            "Filename": cell_filename_list,
            "Condition": cell_condition_list,
            "Cell_Area": cell_area_list,
            "Compartement_density": cell_density_list
        })

        if 'cells_df' not in locals():
            cells_df = cell_df_temp
        else:
            cells_df = pd.concat([cells_df, cell_df_temp], ignore_index=True)

# ---------------- SAVE RESULTS ----------------
particle_df.to_csv(f"/{output_folder}/raw_compartments_{date}.csv", index=False)
cells_df.to_csv(f"/{output_folder}/raw_cells_{date}.csv", index=False)

print("Processing completed. CSVs exported.")

