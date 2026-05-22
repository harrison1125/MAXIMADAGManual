import os
import re
import csv
import matplotlib.pyplot as plt
import Inputs

def calculate_atomic_percent(ti_mass_fraction, v_mass_fraction):
    """Calculate atomic percentages of Ti and V from mass fractions."""
    v_atomic_weight = 50.9415
    ti_atomic_weight = 47.867

    n_v = v_mass_fraction / v_atomic_weight
    n_ti = ti_mass_fraction / ti_atomic_weight
    n_total = n_v + n_ti

    at_percent_v = (n_v / n_total) * 100
    at_percent_ti = (n_ti / n_total) * 100

    return at_percent_ti, at_percent_v

# === Root directory to search ===
root_dir = Inputs.root_dir

# Regex pattern updated for Ti and V
pattern = re.compile(
    r"SOURCE:\s*scan_point_(\d+)\.mca.*?"
    r"Ti\s+K\s+[\d.eE+-]+\s+[\d.eE+-]+\s+([\d.eE+-]+).*?"
    r"V\s+K\s+[\d.eE+-]+\s+[\d.eE+-]+\s+([\d.eE+-]+)",
    re.DOTALL
)

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.lower().endswith(".txt"):
            full_path = os.path.join(dirpath, filename)
            print(f"Processing: {full_path}")

            with open(full_path, 'r') as f:
                data_text = f.read()

            matches = pattern.findall(data_text)
            if not matches:
                continue

            ti_mass = {}
            v_mass = {}
            ti_at_percent = {}
            v_at_percent = {}

            for scan_point, ti_frac, v_frac in matches:
                sp = int(scan_point)
                ti_f, v_f = float(ti_frac), float(v_frac)
                
                ti_mass[sp] = ti_f
                v_mass[sp] = v_f
                
                # Calculate both
                at_ti, at_v = calculate_atomic_percent(ti_f, v_f)
                ti_at_percent[sp] = at_ti
                v_at_percent[sp] = at_v

            sorted_points = sorted(ti_mass.keys())
            ti_at_values = [ti_at_percent[sp] for sp in sorted_points]
            v_at_values = [v_at_percent[sp] for sp in sorted_points]

            # === Plot Results ===
            plt.figure(figsize=(10, 6))
            plt.plot(sorted_points, ti_at_values, marker='^', color='purple', label='Ti At%')
            plt.plot(sorted_points, v_at_values, marker='s', color='orange', label='V At%')
            
            plt.xlabel('Scan Point')
            plt.ylabel('Atomic Percent (%)')
            plt.title(f'Composition Profile: {filename}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            output_png = os.path.join(dirpath, f"{os.path.splitext(filename)[0]}_Composition.png")
            plt.savefig(output_png)
            plt.close()

            # === Write CSV ===
            output_csv = os.path.join(dirpath, f"{os.path.splitext(filename)[0]}_Composition.csv")
            with open(output_csv, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Scan Point", "Ti Mass Frac", "V Mass Frac", "Ti At%", "V At%"])
                for sp in sorted_points:
                    writer.writerow([sp, ti_mass[sp], v_mass[sp], ti_at_percent[sp], v_at_percent[sp]])
