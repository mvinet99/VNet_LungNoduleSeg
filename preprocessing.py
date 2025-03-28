### Preprocess the CT volumes before feeding them into the VNet
#  For each nodule in the UCLA, NLST, and LIDC datasets:
#   1. Resample images to 1.5x1.5x1.5
#   2. Obtain the nodule centroid coordinate - for now from spreadsheet. In the future, from MONAI. 
#      Convert the coordinate to the correct xyz coordinate for trimming
#   3. Trim to a fixed 96x96x16 size centered on the nodule center coordinate, saving as .npy
#   4. Save diameter to a separate folder

import numpy as np
import nibabel as nib
import numpy as np
import torchio as tio
import pandas as pd
from scipy.ndimage import label, center_of_mass
from tqdm import tqdm

def resample_nii(input_path: str,
                 target_spacing: tuple = (1.5, 1.5, 1.5),
                 mode="linear"):
    """
    Resample a nii.gz file to a specified spacing using torchio.

    Parameters:
    - input_path: Path to the input .nii.gz file.
    - target_spacing: Desired spacing for resampling. Default is (1.5, 1.5, 1.5).
    """
    # Load the nii.gz file using torchio
    subject = tio.Subject(img=tio.ScalarImage(input_path))
    resampler = tio.Resample(target=target_spacing, image_interpolation=mode)
    resampled_subject = resampler(subject)
    return resampled_subject.img  

def find_object_centroid(mask):
    # Ensure mask is binary
    mask = mask.get_fdata()

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

def trim_nifti(img, center, box_size):
    """
    Trims a NIfTI image around a given center point to a fixed box size.

    Parameters:
        nifti_path (str): Path to the input NIfTI file.
        center (tuple): The center point (z, y, x) in voxel coordinates.
        box_size (tuple): The size of the box (depth, height, width).
        output_path (str): Path to save the trimmed NIfTI file.
    """

    # Load the NIfTI image
    data = img.data.numpy().squeeze(axis=0)
    # affine = img.affine
    # header = img.header
    
    # Extract shape and center point
    zc, yc, xc = center
    dz, dy, dx = box_size
    z_size, y_size, x_size = data.shape

    # Compute trimming indices while ensuring bounds
    z_start, z_end = max(0, zc - dz // 2), min(z_size, zc + dz // 2)
    y_start, y_end = max(0, yc - dy // 2), min(y_size, yc + dy // 2)
    x_start, x_end = max(0, xc - dx // 2), min(x_size, xc + dx // 2)

    # Trim the image data
    trimmed_data = data[z_start:z_end, y_start:y_end, x_start:x_end]

    return trimmed_data

def cartesian_to_voxel(img, cartesian_coords, coord_system):
    """
    Converts real-world Cartesian coordinates to voxel indices based on the given NIfTI file.
    
    Parameters:
    - img: the image. 
    - cartesian_coords (numpy array): Nx3 array of coordinates in real-world space.
    - coord_system (str): The coordinate system of the input (e.g., 'RAI', 'LAS', 'LPI', 'LPS').
    
    Returns:
    - voxel_indices (numpy array): Nx3 array of voxel indices.
    """

    # Load NIfTI file
    image_orientation = "LAS"  # Set target orientation
    affine = img.affine  # Get affine transformation matrix
    inv_affine = np.linalg.inv(affine)  # Compute inverse
    cartesian_coords = np.array([[cartesian_coords[0]],[cartesian_coords[1]],[cartesian_coords[2]]]).T
    # Convert input coordinates to match NIfTI orientation
    cartesian_coords_adjusted = cartesian_coords.copy()

    # Define how to flip coordinates to match NIfTI orientation
    flip_x = (coord_system[0] != image_orientation[0])  # Flip X if different
    flip_y = (coord_system[1] != image_orientation[1])  # Flip Y if different
    flip_z = (coord_system[2] != image_orientation[2])  # Flip Z if different

    if flip_x:
        cartesian_coords_adjusted[:, 0] *= -1
    if flip_y:
        cartesian_coords_adjusted[:, 1] *= -1
    if flip_z:
        cartesian_coords_adjusted[:, 2] *= -1

    # Convert to homogeneous coordinates
    homogeneous_coords = np.hstack([cartesian_coords_adjusted, np.ones((cartesian_coords.shape[0], 1))])

    # Apply inverse affine transformation
    voxel_coords = inv_affine @ homogeneous_coords.T

    # Round and convert to int
    voxel_indices = np.round(voxel_coords[:3, :].T).astype(int)
    
    return voxel_indices

def resample_coordinate(coord, nii_path, new_spacing):
    """
    Adjusts a coordinate (x, y, z) from the original image space to the new resampled space.
    
    Args:
        coord (tuple): Original coordinate (x, y, z)
        nii_path (str): Path to the original NIfTI file
        new_spacing (tuple): New voxel spacing (dx, dy, dz)

    Returns:
        tuple: Updated coordinate (x', y', z') in the resampled image space
    """
    # Load the NIfTI file
    img = nib.load(nii_path)

    # Get original spacing from NIfTI header
    original_spacing = img.header.get_zooms()  # (dx, dy, dz)

    # Compute the new coordinate
    scale_factors = np.array(original_spacing) / np.array(new_spacing)
    new_coord = np.array(coord) * scale_factors

    return tuple(new_coord)

# Read all nodules to run
nodule_csv = pd.read_csv('/radraid2/mvinet/VNet_Lung_Nodule_Segmentation/datasheets/images_and_segpaths_2024-12-11-checkpoint.csv')
nodule_list = list(nodule_csv['image_path'])
nodule_pids = list(nodule_csv['pid'])
nodule_list_labels = list(nodule_csv['nodule_path'])
nodule_dataset = list(nodule_csv['Dataset'])
4
# Read all UCLA, LIDC, and NLST spreadsheets containing pids, xyz coordinates, and diameter
ucla_csv = pd.read_csv('/radraid2/mvinet/VNet_Lung_Nodule_Segmentation/datasheets/UCLAIDx_Task78_Lesion_info.csv')
ucla_pid = list(ucla_csv['patient_id'])
ucla_diam = list(ucla_csv['diam_long'])

lidc_csv = pd.read_csv('/radraid2/mvinet/VNet_Lung_Nodule_Segmentation/datasheets/LIDC_lesion_info_maybe.csv')
lidc_pid = list(lidc_csv['pid'])
lidc_x = list(lidc_csv['coordX'])
lidc_y = list(lidc_csv['coordY'])
lidc_z = list(lidc_csv['coordZ'])
lidc_paths = list(lidc_csv['image_path'])
lidc_paths_labels = list(lidc_csv['seg_path'])
lidc_diam = list(lidc_csv['diameter'])

nlst_csv = pd.read_csv('/radraid2/mvinet/VNet_Lung_Nodule_Segmentation/datasheets/NLST_merged_lesion_info.csv')
nlst_pid = list(nlst_csv['pid'])
nlst_label_paths = list(nlst_csv['nodule_path'])
nlst_x = list(nlst_csv['coordX'])
nlst_y = list(nlst_csv['coordY'])
nlst_z = list(nlst_csv['coordZ'])
nlst_diam = list(nlst_csv['longest_axial_diameter_(mm)'])

for nodule_path in tqdm(nodule_list, 'Running...'):
    if 'NLST_CT_annotations' in nodule_path or 'Ticket8859' in nodule_path or 'LIDC' in nodule_path:
        skip = 0
    else:
    # 1. Resample images to 1.5x1.5x1.5
        resample_size = (1.5, 1.5, 1.5)
        resampled_img = resample_nii(nodule_path, resample_size) 

        resampled_mask = resample_nii(nodule_list_labels[nodule_list.index(nodule_path)], resample_size)
        
        # 2. Obtain the nodule coordinate from the correct spreadsheet 
        if nodule_dataset[nodule_list.index(nodule_path)] == 'UCLA':
            # For UCLA dataset, there do not exist XYZ coordiantes for nodule centroids. For this, use the label volume to find the centroid

            center = find_object_centroid(nib.load(nodule_list_labels[nodule_list.index(nodule_path)]))
            x = center[0]
            y = center[1]
            z = center[2]
            diam = np.array(ucla_diam[ucla_pid.index(nodule_pids[nodule_list.index(nodule_path)])])
            dname = 'UCLA'
            fname = dname + ucla_pid[ucla_pid.index(nodule_pids[nodule_list.index(nodule_path)])]

            new_coord = resample_coordinate((x,y,z), nodule_path, (1.5,1.5,1.5))
            img = nib.load(nodule_path)
            coord_system = nib.aff2axcodes(img.affine)
            voxelxyz = cartesian_to_voxel(img, new_coord, coord_system)
            x = voxelxyz[0][0]
            y = voxelxyz[0][1]
            z = voxelxyz[0][2]

        elif nodule_dataset[nodule_list.index(nodule_path)] == 'LIDC':
            # For LIDC dataset, use the coordinates in the spreadsheet to convert to the correct nodule centroids (need to use paths)
            lidc_index = lidc_paths_labels.index(nodule_list_labels[nodule_list.index(nodule_path)])
            x = lidc_x[lidc_index]
            y = lidc_y[lidc_index]
            z = lidc_z[lidc_index]
            diam = np.array(lidc_diam[lidc_index])
            dname = 'LIDC'
            fname = dname + lidc_pid[lidc_index]

            new_coord = resample_coordinate((x,y,z), nodule_path, (1.5,1.5,1.5))
            img = nib.load(nodule_path)
            coord_system = nib.aff2axcodes(img.affine)
            voxelxyz = cartesian_to_voxel(img, new_coord, coord_system)
            x = voxelxyz[0][0]
            y = voxelxyz[0][1]
            z = voxelxyz[0][2]

        elif nodule_dataset[nodule_list.index(nodule_path)] == 'NLST':
            # For the NLST dataset, use the coordinates in the spreadsheet to convert to the correct nodule centroids

            nlst_index = nlst_label_paths.index(nodule_list_labels[nodule_list.index(nodule_path)])
            x = nlst_x[nlst_index]
            y = nlst_y[nlst_index]
            z = nlst_z[nlst_index]
            diam = np.array(nlst_diam[nlst_index])
            dname = 'NLST'
            fname = dname + nlst_pid[nlst_index]

            new_coord = resample_coordinate((x,y,z), nodule_path, (1.5,1.5,1.5))
            img = nib.load(nodule_path)
            coord_system = nib.aff2axcodes(img.affine)
            voxelxyz = cartesian_to_voxel(img, new_coord, coord_system)
            x = voxelxyz[0][0]
            y = voxelxyz[0][1]
            z = voxelxyz[0][2]
        
        # 3. Trim to a fixed 96x96x16 size centered on the nodule center coordinate, saving as .npy
        trimmed_img = trim_nifti(resampled_img, (x, y, z), (96, 96, 16))
        trimmed_mask = trim_nifti(resampled_mask, (x, y, z), (96, 96, 16))

        # 4. Save diameter, image, and label to a separate folders

        # Save image
        np.save('/radraid2/mvinet/VNet_Lung_Nodule_Segmentation/data/'+dname+'/images/'+fname,trimmed_img)

        # Save mask
        np.save('/radraid2/mvinet/VNet_Lung_Nodule_Segmentation/data/'+dname+'/labels/'+fname,trimmed_mask)

        # Save diameter
        np.save('/radraid2/mvinet/VNet_Lung_Nodule_Segmentation/data/'+dname+'/diameters/'+fname,diam)
