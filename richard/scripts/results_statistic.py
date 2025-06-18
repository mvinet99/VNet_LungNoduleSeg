import numpy as np
import pandas as pd
import os
from richard.src.test.test import select_checkpoint_interactive

IMAGES_DIR = '/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/data/splits/val/images_2D_2axis'
MASKS_DIR = '/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/data/splits/val/masks_2D_2axis'

checkpoint = select_checkpoint_interactive('/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/checkpoints').split('.')[0]
PRED_MASKS_DIR = f'/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/results/{checkpoint}/predicted_masks'

images_filenames = os.listdir(IMAGES_DIR)
masks_filenames = os.listdir(MASKS_DIR)
pred_masks_patient_dirs = os.listdir(PRED_MASKS_DIR)
pred_masks_filenames = [fname for patient_dir in pred_masks_patient_dirs for fname in os.listdir(os.path.join(PRED_MASKS_DIR, patient_dir))]

images_filenames = np.array(sorted(images_filenames))
masks_filenames = np.array(sorted(masks_filenames))
pred_masks_filenames = np.array(sorted(pred_masks_filenames))

# check if the number of images, masks and pred_masks are the same
if len(images_filenames) != len(masks_filenames) or len(images_filenames) != len(pred_masks_filenames):
    raise ValueError(f'The number of images, masks and pred_masks are not the same. {len(images_filenames)}, {len(masks_filenames)}, {len(pred_masks_filenames)}')

pids = np.array(np.unique([filename.split('_')[0] for filename in images_filenames]))

def get_all_patient_slices(pid:str, filenames:np.ndarray[str]) -> np.ndarray[str]:
    is_pid = np.char.startswith(filenames, pid + '_')
    all_slices = filenames[is_pid]
    return all_slices

def load_slice(slice_paths:np.ndarray[str]) -> np.ndarray:
    slices = np.array([np.load(path) for path in slice_paths])
    return slices

def get_pos_voxels(mask_slices:np.ndarray) -> int:
    return np.sum(mask_slices)

def get_dice_score(true_masks:np.ndarray, pred_masks:np.ndarray) -> float:
    dice_score = 2 * np.sum(true_masks * pred_masks) / (np.sum(true_masks) + np.sum(pred_masks))
    return dice_score

results_list = []
prev_pid = None
prev_num_slices = 0
prev_pos_voxels = 0
prev_dice_score = 0

for pid in pids:
    img_slices_names = get_all_patient_slices(pid, images_filenames)
    mask_slices_names = get_all_patient_slices(pid, masks_filenames)
    pred_mask_slices_names = get_all_patient_slices(pid, pred_masks_filenames)

    if pid == 'UCLA8':
        print(img_slices_names)
        print(mask_slices_names)
        print(pred_mask_slices_names)

    img_slice_paths = [os.path.join(IMAGES_DIR, slice) for slice in img_slices_names]
    mask_slice_paths = [os.path.join(MASKS_DIR, slice) for slice in mask_slices_names]
    pred_mask_slice_paths = [os.path.join(PRED_MASKS_DIR, slice.split('_')[0], slice) for slice in pred_mask_slices_names]

    img_slices = load_slice(img_slice_paths)
    mask_slices = load_slice(mask_slice_paths)
    pred_mask_slices = load_slice(pred_mask_slice_paths)

    num_slices = len(img_slices)
    pos_voxels = get_pos_voxels(mask_slices)
    dice_score = get_dice_score(mask_slices, pred_mask_slices)
    
    if prev_num_slices != num_slices or prev_pos_voxels != pos_voxels or prev_dice_score != dice_score:
        results_list.append({'pid': pid, 'num_slices': num_slices, 'pos_voxels': pos_voxels, 'dice_score': dice_score})
    else:
        print(f"{pid} is likely a duplicate of {prev_pid}")

    prev_pid = pid
    prev_num_slices = num_slices
    prev_pos_voxels = pos_voxels
    prev_dice_score = dice_score

df = pd.DataFrame(results_list)

df.to_csv(f'/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/results/results_statistic_{checkpoint}.csv', index=False)
print(df)