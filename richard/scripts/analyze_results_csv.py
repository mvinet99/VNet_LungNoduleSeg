import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

checkpoint = 'final_checkpoint_25-04-24_00:05:13'
df = pd.read_csv(f'/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/results/results_statistic_{checkpoint}.csv')
print(df)

# Plot histogram of dice score
plt.figure()
plt.hist(df['dice_score'], bins=20, edgecolor='black')
plt.title('Histogram of Dice Score')
plt.xlabel('Dice Score')
plt.ylabel('Frequency')
plt.show()
plt.savefig(f'/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/results/results_statistic_{checkpoint}_histograms_dice.png')

# Plot histogram of number of slices
plt.figure()
plt.hist(df['num_slices'], bins=20, edgecolor='black')
plt.title('Histogram of Number of Slices')
plt.xlabel('Number of Slices')
plt.ylabel('Frequency')
plt.savefig(f'/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/results/results_statistic_{checkpoint}_histograms_slices.png')
# --- Histograms of pos_voxels based on Dice Score threshold ---

# Filter data based on Dice score
df_dice_gt_05 = df[df['dice_score'] > 0.5]
df_dice_lt_05 = df[df['dice_score'] < 0.5]

# Plot histogram of number of positive voxels (Dice > 0.5, Log Scale)
plt.figure()
plt.hist(df_dice_gt_05['pos_voxels'], bins=20, edgecolor='black')
plt.title('Histogram of Positive Voxels (Dice > 0.5, Log Scale)')
plt.xlabel('Number of Positive Voxels (Log Scale)')
plt.ylabel('Frequency')
plt.savefig(f'/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/results/results_statistic_{checkpoint}_histograms_positives_dice_gt0.5_log.png')

# Plot histogram of number of positive voxels (Dice < 0.5, Log Scale)
plt.figure()
plt.hist(df_dice_lt_05['pos_voxels'], bins=20, edgecolor='black')
plt.title('Histogram of Positive Voxels (Dice < 0.5, Log Scale)')
plt.xlabel('Number of Positive Voxels (Log Scale)')
plt.ylabel('Frequency')
plt.savefig(f'/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/results/results_statistic_{checkpoint}_histograms_positives_dice_lt0.5_log.png')

# Plot dice score vs positive voxels (Log Scale)
plt.figure()
x_log = df['pos_voxels']
y_log = df['dice_score']
plt.scatter(x_log, y_log, alpha=0.5)
plt.xscale('log') # Add log scale for x-axis
plt.title('Dice Score vs Number of Positive Voxels (Log Scale)')
plt.xlabel('Number of Positive Voxels (Log Scale)')
plt.ylabel('Dice Score')
plt.savefig(f'/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/results/results_statistic_{checkpoint}_dice_vs_positives_log.png') # Updated filename

# Plot dice score vs positive voxels (Log Scale)
plt.figure()
x_lin = df['pos_voxels']
y_lin = df['dice_score']
plt.scatter(x_lin, y_lin, alpha=0.5)
# Add trend line
m, b = np.polyfit(x_lin, y_lin, 1)
plt.plot(x_lin, m*x_lin + b, color='red')
plt.title('Dice Score vs Number of Positive Voxels') # Title should be linear
plt.xlabel('Number of Positive Voxels')
plt.ylabel('Dice Score')
plt.savefig(f'/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/results/results_statistic_{checkpoint}_dice_vs_positives.png') # Updated filename

# --- Plots split by pos_voxels = 5000 ---

# Filter data
df_lt_20k = df[df['pos_voxels'] < 5000]
df_ge_20k = df[df['pos_voxels'] >= 5000]


# Plot dice score vs positive voxels (< 20000)
plt.figure()
x_lt = df_lt_20k['pos_voxels']
y_lt = df_lt_20k['dice_score']
plt.scatter(x_lt, y_lt, alpha=0.5)
# Add trend line
m, b = np.polyfit(x_lt, y_lt, 1)
plt.plot(x_lt, m*x_lt + b, color='red')
plt.title('Dice Score vs Number of Positive Voxels (< 5000)') # Corrected title threshold
plt.xlabel('Number of Positive Voxels')
plt.ylabel('Dice Score')
plt.savefig(f'/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/results/results_statistic_{checkpoint}_dice_vs_positives_lt5k.png')

# Plot dice score vs positive voxels (>= 20000)
plt.figure()
x_ge = df_ge_20k['pos_voxels']
y_ge = df_ge_20k['dice_score']
plt.scatter(x_ge, y_ge, alpha=0.5)
# Add trend line
m, b = np.polyfit(x_ge, y_ge, 1)
plt.plot(x_ge, m*x_ge + b, color='red')
plt.title('Dice Score vs Number of Positive Voxels (>= 5000)') # Corrected title threshold
plt.xlabel('Number of Positive Voxels')
plt.ylabel('Dice Score')
plt.savefig(f'/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/results/results_statistic_{checkpoint}_dice_vs_positives_ge5k.png')

