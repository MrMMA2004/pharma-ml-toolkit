# Molecular-dynamics metric plots

In PharmaML Studio, open **Molecular Dynamics** from the sidebar and choose a
study folder. Select a ligand and metric, edit the axis labels to match the
input units, and select **Plot metric**. To compare two metrics, choose the
second metric and select **Plot correlation**. Export the current figure as
PNG, SVG or PDF. Plot previews are static; a study load runs in a background
worker. The Studio plot controls require the optional `reports` dependencies.

Use the optional `reports` extra to plot parsed GROMACS metric exports. For
example, a study directory may contain one subfolder per ligand, each with
files such as `ligand_rmsd.xvg` and `ligand_sasa.xvg`.

```python
from pharmaml.md import build_study
from pharmaml.viz import close_figure, plot_md_correlation, plot_md_series, save_figure

study = build_study("my_md_study")
fig = plot_md_series(
    study.metric_series("ligand_rmsd"),
    title="Ligand RMSD by simulation",
    x_label="Time (ns)",  # Replace with the actual units in your data.
    y_label="RMSD (nm)",
)
save_figure(fig, "rmsd.svg")  # Also supports PNG and PDF.
close_figure(fig)

ligand = study.ligand("ligand_A")
# Select metrics sampled on the same time axis.
fig = plot_md_correlation({
    "RMSD": ligand.series["ligand_rmsd"],
    "SASA": ligand.series["ligand_sasa"],
})
save_figure(fig, "correlation.png")
close_figure(fig)
```

`plot_md_series` also accepts a single `XvgSeries`; `plot_md_correlation`
can accept a `LigandResult` directly if all its metrics share the same axis.
These functions do not infer units, resample, or smooth input data. Confirm
axis labels and units against the input files and any preprocessing performed
by `build_study`: its loader conditionally converts time and energy and may
smooth hydrogen-bond data. For cross-ligand plots, ensure the series use
compatible units. Correlations require identical, finite axes and finite
values; mismatches raise an error instead of silently pairing unrelated
samples. Constant-series correlations are undefined and shown as `N/A`.
Pearson correlation is descriptive and does not establish causation.

Install plotting dependencies with `pip install 'pharmaml[reports]'`.
