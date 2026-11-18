/**
 * The nine-workspace registry — part11 §10.1.
 *
 * The left-to-right order is EXACT and never changes; Introduction is
 * selected at startup and the initial page is Introduction ONLY (the
 * old nine-card dashboard must not return as the home page). The
 * existing detailed PySide6 pages reorganize as workflow stages /
 * tabs / advanced panels inside these workspaces in later campaign
 * rows; the queued rows are honest plans, not claims.
 */

export type WorkspaceId =
  | "introduction"
  | "datasets"
  | "machine-learning"
  | "deep-learning"
  | "generative-ai"
  | "molecular-docking"
  | "pkpd"
  | "molecular-dynamics"
  | "visualization";

export interface WorkspaceEntry {
  readonly id: WorkspaceId;
  /** The navigation label (the exact §10.1 wording). */
  readonly label: string;
  /** One-line purpose (the Introduction page summarizes these). */
  readonly summary: string;
  /**
   * Build state of the workspace row, stated honestly: `shell` means
   * only the workspace route exists; `workflow` means the §11
   * sequential workflow is connected to the real backend API.
   */
  readonly state: "shell" | "workflow";
}

export const WORKSPACES: readonly WorkspaceEntry[] = [
  {
    id: "introduction",
    label: "Introduction",
    summary:
      "What PharmaML Studio is, the scientific areas it covers, and the offline local-backend architecture.",
    state: "workflow",
  },
  {
    id: "datasets",
    label: "Datasets",
    summary:
      "Import → Inspect → Validate → Clean/Standardize → Descriptor/Fingerprint Generation → Feature Selection → Split/Leakage Audit → Visualize → Export.",
    state: "workflow",
  },
  {
    id: "machine-learning",
    label: "Machine Learning",
    summary:
      "Select Dataset → Target → Features → Preprocess → Configure Split/CV → Algorithm → Hyperparameters → Train → Evaluate → Applicability Domain → Explainability → Export.",
    state: "workflow",
  },
  {
    id: "deep-learning",
    label: "Deep Learning",
    summary:
      "Select Dataset → Validate → Preprocess → Architecture → Configure → Train with live loss/val curves → Inspect Predictions → Save Checkpoint.",
    state: "workflow",
  },
  {
    id: "generative-ai",
    label: "Generative AI",
    summary:
      "Select Method → Starting Molecules → Constraints/Objectives → Generate → Validate → Properties → Filter → Rank → Visualize → Export.",
    state: "workflow",
  },
  {
    id: "molecular-docking",
    label: "Molecular Docking",
    summary:
      "Ligands → Prepare → Receptor → Binding Site/Grid → Engine → Dock → Poses → Interactions → Export.",
    state: "workflow",
  },
  {
    id: "pkpd",
    label: "PK/PD",
    summary:
      "Import Data → Inspect → Model → Parameters → Fit/Simulate → Diagnostics → Exposure/Response → Export.",
    state: "shell",
  },
  {
    id: "molecular-dynamics",
    label: "Molecular Dynamics Analysis",
    summary:
      "Load Topology/Trajectory → Validate → Select Analysis → Execute → Inspect → Visualize → Export. Trajectory analysis, distinct from simulation.",
    state: "shell",
  },
  {
    id: "visualization",
    label: "Visualization",
    summary:
      "Select Dataset/Analysis Result → Visualization → Variables → Generate → Inspect → Compare → Export.",
    state: "shell",
  },
] as const;

/** The exact §10.1 order as plain labels (the drift-guarded contract). */
export const WORKSPACE_ORDER: readonly string[] = WORKSPACES.map(
  (w) => w.label,
);

export const INITIAL_WORKSPACE: WorkspaceId = "introduction";
