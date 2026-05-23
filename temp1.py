import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
SOURCE_DIR = Path("/Users/hpark108/Desktop/Piyush Rohit Solutionized CuTi Final")
OUTPUT_DIR = Path("/Users/hpark108/Desktop/all_titanium_heatmaps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 75 coordinate pairs corresponding to a 5x5 macro-grid (3 sub-scans per grid point)
COORDINATES = np.array([
    [-6.5, -12], [-6.0, -12], [-5.5, -12], [-0.5, -12], [0.0, -12], [0.5, -12], [5.5, -12], [6.0, -12], [6.5, -12], [11.5, -12], [12.0, -12], [12.5, -12], [17.5, -12], [18.0, -12], [18.5, -12],
    [-6.5, -6],  [-6.0, -6],  [-5.5, -6],  [-0.5, -6],  [0.0, -6],  [0.5, -6],  [5.5, -6],  [6.0, -6],  [6.5, -6],  [11.5, -6],  [12.0, -6],  [12.5, -6],  [17.5, -6],  [18.0, -6],  [18.5, -6],
    [-6.5, 0],   [-6.0, 0],   [-5.5, 0],   [-0.5, 0],   [0.0, 0],   [0.5, 0],   [5.5, 0],   [6.0, 0],   [6.5, 0],   [11.5, 0],   [12.0, 0],   [12.5, 0],   [17.5, 0],   [18.0, 0],   [18.5, 0],
    [-6.5, 6],   [-6.0, 6],   [-5.5, 6],   [-0.5, 6],   [0.0, 6],   [0.5, 6],   [5.5, 6],   [6.0, 6],   [6.5, 6],   [11.5, 6],   [12.0, 6],   [12.5, 6],   [17.5, 6],   [18.0, 6],   [18.5, 6],
    [-6.5, 12],  [-6.0, 12],  [-5.5, 12],  [-0.5, 12],  [0.0, 12],  [0.5, 12],  [5.5, 12],  [6.0, 12],  [6.5, 12],  [11.5, 12],  [12.0, 12],  [12.5, 12],  [17.5, 12],  [18.0, 12],  [18.5, 12]
])

# Collapse the 3 micro-scans into a single center point for the macro-grid coordinate
X_coords_averaged = COORDINATES[::3, 0] + 0.5  
Y_coords_averaged = COORDINATES[::3, 1]

def generate_heatmap(csv_path, run_name):
    try:
        # Read the clean CSV normally
        df = pd.read_csv(csv_path)
        
        # Pull the 4th column (index 3) regardless of any header string variations
        values = df.iloc[:, 3].values
        
        # Validation gate to ensure we are dealing with full grid files
        if len(values) != 75:
            print(f"  Skipping {run_name}: Expected 75 data points, found {len(values)}.")
            return

        # Reshape to 25 rows (positions) by 3 columns (sub-scans) and compute the row means
        averaged_values = values.reshape(-1, 3).mean(axis=1)

        # Plot Setup
        plt.figure(figsize=(9, 7))
        
        # Render scatter plot using large square markers ('s') to generate the grid appearance
        sc = plt.scatter(X_coords_averaged, Y_coords_averaged, c=averaged_values, 
                         cmap='cividis', marker='s', s=1400, edgecolors='black')
        
        # overlay numeric values cleanly centered on each block
        for x, y, val in zip(X_coords_averaged, Y_coords_averaged, averaged_values):
            plt.text(x, y, f"{val:.2f}", ha='center', va='center', 
                     color='white' if val < averaged_values.mean() else 'black', 
                     fontweight='bold', fontsize=8)
            
        plt.colorbar(sc, label="Averaged Ti Atomic Percent")
        plt.xlabel("X Coordinate (mm)")
        plt.ylabel("Y Coordinate (mm)")
        plt.title(f"Ti Composition Grid Map\n{run_name}", fontsize=11, fontweight='bold')
        
        plt.xlim(-10, 22)
        plt.ylim(-16, 16)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        output_file = OUTPUT_DIR / f"{run_name}_TiHeatmap.png"
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"  Successfully generated: {output_file.name}")

    except Exception as e:
        print(f"  Error processing {run_name}: {str(e)}")

# --- Execution Loop ---
print(f"Scanning for 'TiAtomicPercent.csv' tracks in: {SOURCE_DIR}\n" + "="*60)

for filepath in SOURCE_DIR.rglob("TiAtomicPercent.csv"):
    run_folder_id = filepath.parent.parent.name
    print(f"Processing folder: {run_folder_id}")
    generate_heatmap(filepath, run_folder_id)

print("="*60 + f"\nExecution Complete! Heatmaps exported to: {OUTPUT_DIR}")
