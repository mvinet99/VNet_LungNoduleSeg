from __future__ import print_function, division
from pathlib import Path
import os
import SimpleITK as sitk
import numpy as np
from glob import glob
from scipy.ndimage import label, center_of_mass
import pandas as pd
from tqdm import tqdm 

def find_object_centroid(mask):
    # Ensure mask is binary

    mask = (mask == 1)

    # Label connected components
    labeled_array, num_features = label(mask)

    if num_features == 0:
        return None  # No objects found

    # Get the largest connected component
    unique, counts = np.unique(labeled_array, return_counts=True)
    largest_label = unique[1:][np.argmax(counts[1:])]  # Ignore background (label 0)

    # Compute centroid of the largest object
    centroid = center_of_mass(mask, labeled_array, largest_label)
    
    return centroid


def crop_around_center(image, mask, center, target_shape=(64, 64, 64)):
    """
    Crop the image and mask around the center coordinate, ensuring the result has the target shape.
    """
    center_z, center_y, center_x = center

    # Calculate the crop bounds
    z_start = int(center_z - target_shape[0] // 2)
    z_end = int(center_z + target_shape[0] // 2)
    y_start = int(center_y - target_shape[1] // 2)
    y_end = int(center_y + target_shape[1] // 2)
    x_start = int(center_x - target_shape[2] // 2)
    x_end = int(center_x + target_shape[2] // 2)

    # Ensure the crop is within bounds of the image
    z_start = max(z_start, 0)
    z_end = min(z_end, image.shape[0])
    y_start = max(y_start, 0)
    y_end = min(y_end, image.shape[1])
    x_start = max(x_start, 0)
    x_end = min(x_end, image.shape[2])

    # Crop the image and mask
    cropped_image = image[z_start:z_end, y_start:y_end, x_start:x_end]
    cropped_mask = mask[z_start:z_end, y_start:y_end, x_start:x_end]

    # If the cropped region is smaller than the target shape, pad with zeros
    if cropped_image.shape != target_shape:
        padding = [(0, max(target_shape[i] - cropped_image.shape[i], 0)) for i in range(3)]
        cropped_image = np.pad(cropped_image, padding, mode='constant', constant_values=0)
        cropped_mask = np.pad(cropped_mask, padding, mode='constant', constant_values=0)

    return cropped_image, cropped_mask


# Some helper functions
def make_mask(center, diam, z, width, height, spacing, origin):
    '''
    Center : centers of circles px -- list of coordinates x,y,z
    diam : diameters of circles px -- diameter
    widthXheight : pixel dim of image
    spacing = mm/px conversion rate np array x,y,z
    origin = x,y,z mm np.array
    z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height, width])  # 0's everywhere except nodule swapping x,y to match img
    v_center = (center-origin)/spacing
    v_diam = int(diam/spacing[0]+5)
    v_xmin = np.max([0, int(v_center[0]-v_diam)-5])
    v_xmax = np.min([width-1, int(v_center[0]+v_diam)+5])
    v_ymin = np.max([0, int(v_center[1]-v_diam)-5])
    v_ymax = np.min([height-1, int(v_center[1]+v_diam)+5])

    v_xrange = range(v_xmin, v_xmax+1)
    v_yrange = range(v_ymin, v_ymax+1)

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x, p_y, z])) <= diam:
                mask[int((p_y-origin[1])/spacing[1]), int((p_x-origin[0])/spacing[0])] = 1.0
    return mask

def matrix2int16(matrix):
    ''' 
    matrix must be a numpy array NXN
    Returns uint16 version
    '''
    m_min = np.min(matrix)
    m_max = np.max(matrix)
    matrix = matrix - m_min
    return np.array(np.rint((matrix - m_min) / float(m_max - m_min) * 65535.0), dtype=np.uint16)


############
paths = ["data/subset0/subset0/","data/subset1/subset1/","data/subset2/subset2/","data/subset3/subset3/","data/subset4/subset4/","data/subset5/subset5/","data/subset6/subset6/","data/subset7/subset7/","data/subset8/subset8/","data/subset9/subset9/"]
for path in paths:
    # Getting list of image files
    luna_path = "F:/LUNA16/"
    luna_subset_path = luna_path + path
    output_path_image = "F:/LUNA16/Volumes_modified/images/"
    output_path_mask = "F:/LUNA16/Volumes_modified/masks/"
    file_list = glob(luna_subset_path + "*.mhd")

    #####################
    #
    # Helper function to get rows in data frame associated 
    # with each file
    def get_filename(file_list, case):
        for f in file_list:
            if case in f:
                return f


    # The locations of the nodes
    df_node = pd.read_csv(luna_path + "annotations.csv")
    df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
    df_node = df_node.dropna()

    #####
    #
    # Looping over the image files
    #
    for fcount, img_file in enumerate(tqdm(file_list)):
        try:
            mini_df = df_node[df_node["file"] == img_file]  # get all nodules associated with file
            if mini_df.shape[0] > 0:  # some files may not have a nodule--skipping those
                # load the data once
                itk_img = sitk.ReadImage(img_file)
                img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z, y, x (notice the ordering)
                num_z, height, width = img_array.shape        # heightXwidth constitute the transverse plane
                origin = np.array(itk_img.GetOrigin())        # x, y, z  Origin in world coordinates (mm)
                spacing = np.array(itk_img.GetSpacing())      # spacing of voxels in world coor. (mm)

                # go through all nodes
                for node_idx, cur_row in mini_df.iterrows():
                    node_x = cur_row["coordX"]
                    node_y = cur_row["coordY"]
                    node_z = cur_row["coordZ"]
                    diam = cur_row["diameter_mm"]

                    # Initialize arrays to hold the 64 slices
                    imgs = np.ndarray([64, height, width], dtype=np.float32)
                    masks = np.ndarray([64, height, width], dtype=np.uint8)
                    center = np.array([node_x, node_y, node_z])  # nodule center
                    v_center = np.rint((center - origin) / spacing)  # nodule center in voxel space (still x, y, z ordering)

                    # Calculate the slice indices for the region centered around the nodule center
                    start_z = int(v_center[2]) - 32  # 32 slices before the center
                    end_z = int(v_center[2]) + 32    # 32 slices after the center

                    # Ensure we don't go out of bounds (clip the start and end indices to the valid range)
                    start_z = max(start_z, 0)
                    end_z = min(end_z, num_z)

                    # Loop through and select the 64 slices (or fewer if near the image boundaries)
                    # Loop through and select the 64 slices (or fewer if near the image boundaries)
                    z_range = range(start_z, end_z)
                    if len(z_range) < 64:
                        # If there are fewer than 64 slices, we pad with the first and last slice
                        padding_before = (64 - len(z_range)) // 2
                        padding_after = 64 - len(z_range) - padding_before
                        z_range = list(range(start_z - padding_before, start_z)) + list(z_range) + list(range(end_z, end_z + padding_after))

                    # Now, extract the images and masks for the selected slices
                    for i, i_z in enumerate(z_range):
                        # Check if the index is out of bounds and pad with zeros if necessary
                        if i_z < 0 or i_z >= num_z:
                            # Padding with zero images and masks
                            masks[i] = np.zeros([height, width], dtype=np.uint8)
                            imgs[i] = np.zeros([height, width], dtype=np.float32)
                        else:
                            mask = make_mask(center, diam, i_z * spacing[2] + origin[2], width, height, spacing, origin)
                            masks[i] = mask
                            imgs[i] = img_array[i_z]


                    centroid = find_object_centroid(masks)
                    cropped_image, cropped_mask = crop_around_center(imgs, masks, centroid, target_shape=(64, 64, 64))

                    # Save the centered images and masks
                    np.save(Path(os.path.join(output_path_image, "images_%04d_%04d.npy" % (fcount, node_idx))), cropped_image)
                    np.save(Path(os.path.join(output_path_mask, "masks_%04d_%04d.npy" % (fcount, node_idx))), cropped_mask)
        except:
            skip = 0