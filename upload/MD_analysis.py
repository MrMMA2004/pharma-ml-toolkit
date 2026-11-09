
import os 
import numpy as np
import matplotlib.pyplot as plt
import csv

KJ_TO_KCAL = 1 / 4.184
# PS_TO_NS = 1 / 1000.0

# -------------------------------------------------------------------
# Simple parser for GROMACS .xvg files
# -------------------------------------------------------------------
def parse_xvg(filename):
    x, y = [], []
    try:
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(('#', '@')):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x.append(float(parts[0]))
                        y.append(float(parts[1]))
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return np.array([]), np.array([])
    return np.array(x), np.array(y)

# -------------------------------------------------------------------
# Parser for .ndx (or generic text) HB-number files
# -------------------------------------------------------------------
def parse_ndx_hbnum(filename):
    rows = []
    try:
        with open(filename) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith(('#', '@', '[', ']')):
                    continue
                parts = s.split()
                nums = []
                for p in parts:
                    try:
                        nums.append(float(p))
                    except ValueError:
                        nums.append(np.nan)
                if any(not np.isnan(v) for v in nums):
                    rows.append(nums)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return np.array([]), np.array([])

    if not rows:
        return np.array([]), np.array([])

    maxcols = max(len(r) for r in rows)
    data = np.full((len(rows), maxcols), np.nan)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            if not np.isnan(v):
                data[i, j] = v

    time_col = None
    for j in range(data.shape[1]):
        col = data[:, j]
        if np.all(np.isfinite(col)):
            diffs = np.diff(col)
            if np.all(diffs >= -1e-8) and np.ptp(col) > 0:
                time_col = j
                break

    hb_col = None
    for j in range(data.shape[1]):
        if j == time_col:
            continue
        col = data[:, j]
        if np.all(np.isfinite(col)):
            residual = np.abs(col - np.round(col))
            if np.all(residual < 1e-6) and np.nanmean(np.abs(col)) < 1000:
                hb_col = j
                break

    if time_col is None:
        time_col = 0
    if hb_col is None:
        hb_col = 1 if data.shape[1] > 1 else 0

    x = data[:, time_col]
    y = data[:, hb_col]

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(y) > 5:
        window = 3
        y = np.convolve(y, np.ones(window)/window, mode='same')

    return np.array(x), np.array(y)

# -------------------------------------------------------------------
# Expected files & titles
# -------------------------------------------------------------------
expected_files = {
    "ligand_rmsd.xvg": "ligand_rmsd.jpeg",
    "protein_backbone_rmsd.xvg": "protein_backbone_rmsd.jpeg",
    "ligand_rmsf.xvg": "ligand_rmsf.jpeg",
    "per_residue_rmsf.xvg": "per_residue_rmsf.jpeg",
    "hbnum_intermolecular.xvg": "hbnum_intermolecular.jpeg",
    "hbnum_intramolecular.xvg": "hbnum_intramolecular.jpeg",
    "hbnum_intermolecular.ndx": "hbnum_intermolecular.jpeg",
    "hbnum_intramolecular.ndx": "hbnum_intramolecular.jpeg",
    "protein_backbone_Rg.xvg": "protein_backbone_Rg.jpeg",
    "ligand_sasa.xvg": "ligand_sasa.jpeg",
    "protein_sasa.xvg": "protein_sasa.jpeg",
    "binding_free_energy.xvg": "binding_free_energy.jpeg",
}

plot_titles = {
    "ligand_rmsd.xvg": ("Ligand RMSD", "Time (ns)", "RMSD (Å)"),
    "protein_backbone_rmsd.xvg": ("Protein Backbone RMSD", "Time (ns)", "RMSD (Å)"),
    "ligand_rmsf.xvg": ("Ligand RMSF", "Atom index", "RMSF (Å)"),
    "per_residue_rmsf.xvg": ("Per-residue RMSF", "Residue index", "RMSF (Å)"),
    "hbnum_intermolecular.xvg": ("Intermolecular H-bonds", "Time (ns)", "Number of H-bonds"),
    "hbnum_intramolecular.xvg": ("Intramolecular H-bonds", "Time (ns)", "Number of H-bonds"),
    "hbnum_intermolecular.ndx": ("Intermolecular H-bonds", "Time (ns)", "Number of H-bonds"),
    "hbnum_intramolecular.ndx": ("Intramolecular H-bonds", "Time (ns)", "Number of H-bonds"),
    "protein_backbone_Rg.xvg": ("Protein Backbone Radius of Gyration", "Time (ns)", "Rg (Å)"),
    "ligand_sasa.xvg": ("Ligand SASA", "Time (ns)", "SASA (Å²)"),
    "protein_sasa.xvg": ("Protein SASA", "Time (ns)", "SASA (Å²)"),
    "binding_free_energy.xvg": ("Binding Free Energy", "Time (ns)", "ΔG (kcal/mol)"),
}

color_map = {
    "rmsd": "red", "rmsf": "green", "hbond": "blue",
    "rg": "purple", "sasa": "orange", "energy": "brown"
}

# -------------------------------------------------------------------
# Process one ligand folder
# -------------------------------------------------------------------
def process_folder(folder):
    ligand_name = os.path.basename(folder)
    results_dir = os.path.join(folder, f"{ligand_name}_results")
    os.makedirs(results_dir, exist_ok=True)

    results = {}
    for fname, outname in expected_files.items():
        in_file = os.path.join(folder, fname)
        out_file = os.path.join(results_dir, outname)
        if not os.path.exists(in_file):
            continue

        # Choose color
        if "rmsd" in fname:
            color = color_map["rmsd"]
        elif "rmsf" in fname:
            color = color_map["rmsf"]
        elif "hbnum" in fname:
            color = color_map["hbond"]
        elif "Rg" in fname:
            color = color_map["rg"]
        elif "sasa" in fname:
            color = color_map["sasa"]
        elif "energy" in fname:
            color = color_map["energy"]
        else:
            color = "black"

        # Parse data
        if in_file.endswith(".ndx"):
            x, y = parse_ndx_hbnum(in_file)
        else:
            x, y = parse_xvg(in_file)

        # Convert binding free energy to kcal/mol (only once)
        if "binding_free_energy" in fname:
            y = y * KJ_TO_KCAL

        # Convert time to ns if needed
        xlabel = plot_titles[fname][1]
        ylabel = plot_titles[fname][2]
        if "time" in xlabel.lower() and np.max(x) > 100:
            x = x / 1000.0

        # Plot individual figure
        if len(x) > 0 and len(y) > 0:
            plt.figure(figsize=(6, 4), dpi=300)
            plt.plot(x, y, color=color, lw=1.8)
            plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
            plt.title(plot_titles[fname][0], fontsize=12, fontweight='bold')
            plt.xlabel(xlabel, fontsize=11)
            plt.ylabel(ylabel, fontsize=11)
            plt.tick_params(axis='both', which='major', labelsize=9)
            plt.tight_layout()
            plt.savefig(out_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved {out_file}")

        results[fname] = (x, y)

    # Correlation heatmap per ligand
    try:
        metrics = {}
        for fname, (x, y) in results.items():
            key = fname.replace(".xvg", "").replace(".ndx", "")
            if len(y) > 0:
                metrics[key] = y

        if len(metrics) >= 2:
            min_len = min(len(v) for v in metrics.values())
            data_matrix = np.array([v[:min_len] for v in metrics.values()])
            corr_matrix = np.corrcoef(data_matrix)

            plt.figure(figsize=(6, 5), dpi=300)
            im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            cbar = plt.colorbar(im)
            cbar.set_label("Correlation coefficient", fontsize=12)
            cbar.ax.tick_params(labelsize=8)
            plt.title(f"Correlation Heatmap — {ligand_name}", fontsize=12, fontweight='bold')
            plt.xticks(range(len(metrics)), list(metrics.keys()), rotation=45, ha='right', fontsize=8)
            plt.yticks(range(len(metrics)), list(metrics.keys()), fontsize=8)
            for i in range(len(metrics)):
                for j in range(len(metrics)):
                    plt.text(j, i, f"{corr_matrix[i,j]:.2f}", ha='center', va='center', color='black', fontsize=7)
            plt.tight_layout(pad=0.8)
            out_path = os.path.join(results_dir, "correlation_heatmap.jpeg")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved correlation heatmap: {out_path}")
    except Exception as e:
        print(f"Error generating correlation heatmap for {ligand_name}: {e}")

    return results

# -------------------------------------------------------------------
# Global plots
# -------------------------------------------------------------------
def plot_global(data_dict, out_file, title, xlabel, ylabel):
    plt.figure(figsize=(6,4), dpi=300)
    for label, (x, y, color) in data_dict.items():
        plt.plot(x, y, lw=1.5, label=label, color=color)
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")

def collect_and_plot_global(all_results, global_dir):
    os.makedirs(global_dir, exist_ok=True)
    colors = ["red","green","blue","purple","orange","brown","pink","cyan","magenta","grey"]
    for fname, (title, xlabel, ylabel) in plot_titles.items():
        data_dict = {}
        for i, (folder, results) in enumerate(all_results.items()):
            if not results:
                continue
            if fname in results:
                x, y = results[fname]
                color = colors[i % len(colors)]
                y_plot = y.copy()  # Already in kcal/mol for binding energy
                data_dict[folder] = (x, y_plot, color)
        if data_dict:
            out_file = os.path.join(global_dir, f"global_{expected_files[fname]}")
            plot_global(data_dict, out_file, title, xlabel, ylabel)

# -------------------------------------------------------------------
# Global CSV statistics
# -------------------------------------------------------------------
def save_global_csv(all_results, out_csv):
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Folder","File","Mean","Stdev","Coefficient_of_Variation"])
        for folder, results in all_results.items():
            if not results:
                continue
            for fname, (x, y) in results.items():
                if len(y) > 0:
                    y_to_use = y.copy()  # Already in kcal/mol for binding energy
                    mean, std = np.mean(y_to_use), np.std(y_to_use)
                    cv = std / mean if mean != 0 else np.nan
                    writer.writerow([folder, fname, mean, std, cv])
    print(f"Saved global CSV statistics: {out_csv}")

# -------------------------------------------------------------------
# Multi-panel figures (2 columns per row)
# -------------------------------------------------------------------
def create_multipanel_figures(all_results, global_dir):
    os.makedirs(global_dir, exist_ok=True)
    colors = ["red","green","blue","purple","orange","brown","pink","cyan","magenta","grey"]

    def get_xy(fname):
        data_dict = {}
        for i, (folder, results) in enumerate(all_results.items()):
            if not results:
                continue
            if fname in results:
                x, y = results[fname]
                color = colors[i % len(colors)]
                data_dict[folder] = (x, y, color)
        return data_dict

    figure_defs = {
        "Figure_1_protein_rmsd_rmsf.jpeg": ["protein_backbone_rmsd.xvg", "per_residue_rmsf.xvg"],
        "Figure_2_protein_Rg_hbnum_sasa.jpeg": ["protein_backbone_Rg.xvg", "hbnum_intramolecular.xvg", "protein_sasa.xvg"],
        "Figure_3_ligand_rmsd_hbnum.jpeg": ["ligand_rmsd.xvg", "hbnum_intermolecular.xvg"]
    }

    for fig_name, fnames in figure_defs.items():
        n_subplots = len(fnames)
        n_cols = 2
        n_rows = (n_subplots + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 4*n_rows), dpi=300, constrained_layout=True)
        axes = np.array(axes).reshape(-1)

        for idx, fname in enumerate(fnames):
            ax = axes[idx]
            data_dict = get_xy(fname)
            if not data_dict:
                ax.axis('off')
                continue
            title, xlabel, ylabel = plot_titles[fname]
            for label, (x, y, color) in data_dict.items():
                ax.plot(x, y, lw=1.5, label=label, color=color)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
            ax.tick_params(axis='both', which='major', labelsize=9)
            ax.legend(fontsize=7)

        for ax in axes[n_subplots:]:
            ax.axis('off')

        out_file = os.path.join(global_dir, fig_name)
        plt.savefig(out_file, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved multi-panel figure: {out_file}")

# -------------------------------------------------------------------
# Global multi-panel correlation heatmaps (2 columns per row)
# -------------------------------------------------------------------
def create_global_heatmap_panel(all_results, global_dir):
    os.makedirs(global_dir, exist_ok=True)
    heatmaps = {}
    for folder, results in all_results.items():
        metrics = {}
        for fname, (x, y) in results.items():
            key = fname.replace(".xvg", "").replace(".ndx", "")
            if len(y) > 0:
                metrics[key] = y
        if len(metrics) >= 2:
            min_len = min(len(v) for v in metrics.values())
            data_matrix = np.array([v[:min_len] for v in metrics.values()])
            corr_matrix = np.corrcoef(data_matrix)
            heatmaps[folder] = (list(metrics.keys()), corr_matrix)

    if not heatmaps:
        print("No heatmaps available to create global heatmap panel.")
        return

    n_heatmaps = len(heatmaps)
    n_cols = 2
    n_rows = (n_heatmaps + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows), dpi=300, constrained_layout=True)
    axes = axes.flatten()

    for idx, (folder, (labels, corr_matrix)) in enumerate(heatmaps.items()):
        ax = axes[idx]
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax.set_title(f"{folder} Correlation Heatmap", fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{corr_matrix[i,j]:.2f}", ha='center', va='center', color='black', fontsize=7)

    for ax in axes[n_heatmaps:]:
        ax.axis('off')

    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.03, pad=0.02)
    cbar.set_label("Correlation coefficient", fontsize=12)
    cbar.ax.tick_params(labelsize=8)
    out_file = os.path.join(global_dir, "Global_correlation_heatmaps.jpeg")
    plt.savefig(out_file, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved global multi-panel correlation heatmaps: {out_file}")

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    cwd = os.getcwd()
    subfolders = [f.path for f in os.scandir(cwd) if f.is_dir() and f.name.lower() != "global_plots"]

    all_results = {}

    if not subfolders:
        print("No subfolders found. Running in current directory only...")
        results = process_folder(cwd)
        all_results[os.path.basename(cwd)] = results
    else:
        for folder in subfolders:
            print(f"\nProcessing folder: {folder}")
            results = process_folder(folder)
            ligand_name = os.path.basename(folder)
            all_results[ligand_name] = results

    global_dir = os.path.join(cwd, "Global_plots")
    collect_and_plot_global(all_results, global_dir)
    save_global_csv(all_results, os.path.join(global_dir, "global_summary_stats.csv"))
    create_multipanel_figures(all_results, global_dir)
    create_global_heatmap_panel(all_results, global_dir)

