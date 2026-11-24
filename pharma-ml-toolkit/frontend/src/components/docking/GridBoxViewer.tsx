/**
 * part16 §12 (Phase 9) — the Manual Grid Box's synchronized 3D viewer.
 *
 * The scientific visualization Phase 9 names: the protein structure,
 * the grid box, the binding-site region, and the reference ligand,
 * drawn in ONE scene in the receptor's own coordinate frame.
 *
 * Phase 9's contract, enforced structurally here:
 *
 * * the synchronization rule — the box mesh is re-positioned and
 *   re-scaled from the SAME six numbers the stage-6 dock request
 *   serializes (`center: [centerX, centerY, centerZ]`,
 *   `size: [sizeX, sizeY, sizeZ]`); editing the numeric inputs in the
 *   Manual grid box card updates this view on the next render pass,
 *   and the numeric inputs stay the ONE mutation surface (the viewer
 *   itself is read-only: orbit / zoom / pan can never drift the
 *   coordinates from the submitted request);
 * * the alignment rule — no transform is ever applied to the protein:
 *   the box, the binding-site census, and the ligand all draw in the
 *   receptor's own frame (the pinned REFERENCE_FRAME_NOTE renders
 *   verbatim);
 * * the honesty rules — every atom rendered is a record that actually
 *   arrived (the page's own PDBQT parse; the viewer renders only what
 *   its props carry, never an imagined structure); the spacing note
 *   states that the backend DockRequest carries no spacing field; the
 *   WebGL-unavailable fallback is an honest rendered card, never a
 *   blank canvas;
 * * the required controls — protein / grid-box / ligand /
 *   binding-site visibility toggles, plus the readouts: center,
 *   dimensions, volume, the in-box atom censuses, units (Å), and the
 *   validation warnings;
 * * the memory + resource discipline — every geometry/material/
 *   texture/renderer disposed on unmount, devicePixelRatio capped at
 *   2, the RAF loop pauses offscreen and on document.hidden;
 * * Three.js ONLY for genuine 3D scenes (the house Phase Eighteen
 *   rule) — this is one.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  cpkColor,
  displayRadius,
  type InferredBond,
} from "@/services/molecular3d/pdbqt";
import type { Molecule3DAtom } from "@/components/molecular3d/Molecule3D";
import {
  boxExtents,
  boxVolume,
  countAtomsInBox,
  formatCoord,
  REFERENCE_FRAME_NOTE,
  SPACING_NOTE,
  UNIT_NOTE,
  validateGridBox,
} from "@/services/docking/gridBox";

/** The reference ligand scene the page already parsed (or null). */
export interface GridBoxLigand {
  readonly atoms: readonly Molecule3DAtom[];
  readonly bonds: readonly InferredBond[];
  readonly diagnostics: string | null;
}

export interface GridBoxViewerProps {
  /** The receptor atoms that actually arrived (may be empty). */
  readonly receptorAtoms: readonly Molecule3DAtom[];
  /** The receptor parse diagnostics (skipped/truncated counts). */
  readonly receptorDiagnostics: string | null;
  /** The prepared ligand scene, or null when none arrived yet. */
  readonly ligand: GridBoxLigand | null;
  /** The SAME config the dock request serializes (the manual form's state). */
  readonly config: {
    readonly centerX: number;
    readonly centerY: number;
    readonly centerZ: number;
    readonly sizeX: number;
    readonly sizeY: number;
    readonly sizeZ: number;
  };
}

type WebglSupport = "unchecked" | "available" | "unavailable";

/** Probe WebGL before three ever constructs a renderer. */
function probeWebgl(): WebglSupport {
  if (typeof document === "undefined") {
    return "unavailable";
  }
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
    return gl === null ? "unavailable" : "available";
  } catch {
    return "unavailable";
  }
}

const BOX_COLOR = 0xe6b422;
const BINDING_COLOR = 0x35c4d7;
const AXIS_X_COLOR = 0xe03a3a;
const AXIS_Y_COLOR = 0x35c44f;
const AXIS_Z_COLOR = 0x2f5fe0;
const RECEPTOR_SIZE = 0.85;
const BINDING_SIZE = 1.15;

/** A soft round sprite for the point clouds (squares read as noise). */
function makeRoundSprite(): THREE.Texture | null {
  if (typeof document === "undefined") {
    return null;
  }
  const canvas = document.createElement("canvas");
  canvas.width = 32;
  canvas.height = 32;
  const ctx = canvas.getContext("2d");
  if (ctx === null) {
    return null;
  }
  const gradient = ctx.createRadialGradient(16, 16, 2, 16, 16, 15);
  gradient.addColorStop(0, "rgba(255, 255, 255, 1)");
  gradient.addColorStop(0.7, "rgba(255, 255, 255, 0.85)");
  gradient.addColorStop(1, "rgba(255, 255, 255, 0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 32, 32);
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

interface SceneRefs {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  receptorPoints: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial>;
  bindingPoints: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial>;
  boxEdges: THREE.LineSegments<THREE.EdgesGeometry, THREE.LineBasicMaterial>;
  boxFaces: THREE.Mesh<THREE.BoxGeometry, THREE.MeshBasicMaterial>;
  ligandGroup: THREE.Group;
  axesGroup: THREE.Group;
  sprite: THREE.Texture | null;
  sphereGeometry: THREE.SphereGeometry;
  bondGeometry: THREE.CylinderGeometry;
  bondMaterial: THREE.MeshStandardMaterial;
  raf: number;
  resizeObserver: ResizeObserver;
  intersectionObserver: IntersectionObserver | null;
  visible: boolean;
}

/** The complete teardown — every resource disposed (memory cleanup). */
function disposeScene(refs: SceneRefs): void {
  cancelAnimationFrame(refs.raf);
  refs.intersectionObserver?.disconnect();
  refs.resizeObserver.disconnect();
  refs.controls.dispose();
  for (const child of refs.ligandGroup.children) {
    if (child instanceof THREE.Mesh) {
      child.material.dispose();
    }
  }
  refs.scene.remove(refs.ligandGroup);
  refs.scene.remove(refs.axesGroup);
  refs.axesGroup.traverse((child) => {
    if (child instanceof THREE.Line) {
      child.geometry.dispose();
      (child.material as THREE.Material).dispose();
    }
  });
  refs.receptorPoints.geometry.dispose();
  refs.receptorPoints.material.dispose();
  refs.bindingPoints.geometry.dispose();
  refs.bindingPoints.material.dispose();
  refs.boxEdges.geometry.dispose();
  refs.boxEdges.material.dispose();
  refs.boxFaces.geometry.dispose();
  refs.boxFaces.material.dispose();
  refs.sphereGeometry.dispose();
  refs.bondGeometry.dispose();
  refs.bondMaterial.dispose();
  refs.sprite?.dispose();
  refs.renderer.dispose();
  refs.renderer.domElement.remove();
}

/** One receptor point cloud built from the atoms that arrived. */
function buildCloud(
  atoms: readonly Molecule3DAtom[],
  material: THREE.PointsMaterial,
): THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial> {
  const positions = new Float32Array(atoms.length * 3);
  const colors = new Float32Array(atoms.length * 3);
  const color = new THREE.Color();
  for (let i = 0; i < atoms.length; i += 1) {
    const atom = atoms[i]!;
    positions[i * 3] = atom.x;
    positions[i * 3 + 1] = atom.y;
    positions[i * 3 + 2] = atom.z;
    color.set(cpkColor(atom.element));
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return new THREE.Points(geometry, material);
}

/** The binding-site census cloud: only the receptor atoms inside the box. */
function buildBindingCloud(
  atoms: readonly Molecule3DAtom[],
  material: THREE.PointsMaterial,
  config: GridBoxViewerProps["config"],
): THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial> {
  const extents = boxExtents(config);
  const inside = extents === null
    ? []
    : atoms.filter(
        (a) =>
          a.x >= extents.minX && a.x <= extents.maxX &&
          a.y >= extents.minY && a.y <= extents.maxY &&
          a.z >= extents.minZ && a.z <= extents.maxZ,
      );
  return buildCloud(inside, material);
}

export function GridBoxViewer({
  receptorAtoms,
  receptorDiagnostics,
  ligand,
  config,
}: GridBoxViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<SceneRefs | null>(null);
  const [webgl] = useState<WebglSupport>(() => probeWebgl());
  const [ready, setReady] = useState(false);
  const [sceneError, setSceneError] = useState<string | null>(null);
  const [showProtein, setShowProtein] = useState(true);
  const [showBox, setShowBox] = useState(true);
  const [showLigand, setShowLigand] = useState(true);
  const [showHighlight, setShowHighlight] = useState(true);

  const hasReceptor = receptorAtoms.length > 0;
  const hasLigand = ligand !== null && ligand.atoms.length > 0;

  const receptorInBox = useMemo(
    () => countAtomsInBox(receptorAtoms, config),
    [receptorAtoms, config],
  );
  const ligandInBox = useMemo(
    () => (hasLigand ? countAtomsInBox(ligand.atoms, config) : null),
    [hasLigand, ligand, config],
  );
  const warnings = useMemo(
    () =>
      validateGridBox(config, {
        receptorAtomCount: receptorAtoms.length,
        receptorInBox,
        ligandAtomCount: hasLigand ? ligand.atoms.length : null,
        ligandInBox,
      }),
    [config, receptorAtoms.length, receptorInBox, hasLigand, ligand, ligandInBox],
  );
  const volume = useMemo(() => boxVolume(config), [config]);
  const extents = useMemo(() => boxExtents(config), [config]);

  // the scene lifecycle — built once per webgl availability, torn
  // down completely on unmount (every resource disposed)
  useEffect(() => {
    if (webgl !== "available" || containerRef.current === null) {
      return;
    }
    const container = containerRef.current;
    const height = 380;
    let refs: SceneRefs | null = null;
    let announcedReady = false;
    try {
      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio ?? 1, 2));
      renderer.setSize(container.clientWidth || 640, height);
      container.appendChild(renderer.domElement);
      const camera = new THREE.PerspectiveCamera(
        45,
        (container.clientWidth || 640) / height,
        0.1,
        10000,
      );
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      const sceneThree = new THREE.Scene();
      sceneThree.add(new THREE.AmbientLight(0xffffff, 0.6));
      const key = new THREE.DirectionalLight(0xffffff, 0.9);
      key.position.set(1, 1.6, 0.9);
      sceneThree.add(key);

      const sprite = makeRoundSprite();
      const receptorMaterial = new THREE.PointsMaterial({
        size: RECEPTOR_SIZE,
        sizeAttenuation: true,
        vertexColors: true,
        map: sprite ?? undefined,
        transparent: sprite !== null,
        alphaTest: 0.1,
      });
      const receptorPoints = buildCloud([], receptorMaterial);
      sceneThree.add(receptorPoints);
      const bindingMaterial = new THREE.PointsMaterial({
        size: BINDING_SIZE,
        sizeAttenuation: true,
        color: BINDING_COLOR,
        map: sprite ?? undefined,
        transparent: sprite !== null,
        alphaTest: 0.1,
        depthWrite: false,
      });
      const bindingPoints = buildCloud([], bindingMaterial);
      sceneThree.add(bindingPoints);

      // the grid box: the unit edge wireframe is SCALED to the config —
      // the numeric-input sync lane re-positions/re-scales without a
      // geometry rebuild
      const boxEdges = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
        new THREE.LineBasicMaterial({ color: BOX_COLOR }),
      );
      sceneThree.add(boxEdges);
      const boxFaces = new THREE.Mesh(
        new THREE.BoxGeometry(1, 1, 1),
        new THREE.MeshBasicMaterial({
          color: BOX_COLOR,
          transparent: true,
          opacity: 0.07,
          depthWrite: false,
        }),
      );
      sceneThree.add(boxFaces);

      const ligandGroup = new THREE.Group();
      sceneThree.add(ligandGroup);

      // the coordinate reference frame triad (rebuilt per structure fit)
      const axesGroup = new THREE.Group();
      sceneThree.add(axesGroup);

      const sphereGeometry = new THREE.SphereGeometry(1, 20, 14);
      const bondGeometry = new THREE.CylinderGeometry(0.08, 0.08, 1, 10, 1, true);
      const bondMaterial = new THREE.MeshStandardMaterial({
        color: 0x9aa0a6,
        metalness: 0.15,
        roughness: 0.55,
      });

      refs = {
        renderer,
        scene: sceneThree,
        camera,
        controls,
        receptorPoints,
        bindingPoints,
        boxEdges,
        boxFaces,
        ligandGroup,
        axesGroup,
        sprite,
        sphereGeometry,
        bondGeometry,
        bondMaterial,
        raf: 0,
        resizeObserver: new ResizeObserver(() => {
          const width = container.clientWidth || 640;
          camera.aspect = width / height;
          camera.updateProjectionMatrix();
          renderer.setSize(width, height);
        }),
        intersectionObserver: null,
        visible: true,
      };
      refs.resizeObserver.observe(container);
      const intersectionObserver = new IntersectionObserver((entries) => {
        if (refs === null) {
          return;
        }
        refs.visible = entries.some((e) => e.isIntersecting);
      });
      intersectionObserver.observe(container);
      refs.intersectionObserver = intersectionObserver;

      const renderLoop = () => {
        if (refs === null) {
          return;
        }
        refs.raf = requestAnimationFrame(renderLoop);
        if (!announcedReady) {
          announcedReady = true;
          setReady(true);
          setSceneError(null);
        }
        if (!refs.visible || document.hidden) {
          return;
        }
        refs.controls.update();
        refs.renderer.render(refs.scene, refs.camera);
      };
      refs.raf = requestAnimationFrame(renderLoop);
      sceneRef.current = refs;
    } catch (error) {
      if (refs !== null) {
        disposeScene(refs);
      }
      sceneRef.current = null;
      const message =
        error instanceof Error
          ? error.message
          : "the grid-box scene could not be constructed";
      queueMicrotask(() => {
        setSceneError(message);
        setReady(false);
      });
    }
    return () => {
      if (refs === null) {
        return;
      }
      disposeScene(refs);
      sceneRef.current = null;
      setReady(false);
    };
  }, [webgl]);

  // the numeric-input sync lane (Phase 9's required direction): the
  // box mesh re-positions and re-scales from the SAME config the dock
  // request serializes — every keystroke in the Manual grid box card
  // lands here. The binding-site census cloud follows the box.
  useEffect(() => {
    const refs = sceneRef.current;
    if (refs === null || !ready) {
      return;
    }
    const current = boxExtents(config);
    if (current === null) {
      refs.boxEdges.visible = false;
      refs.boxFaces.visible = false;
    } else {
      refs.boxEdges.visible = showBox;
      refs.boxFaces.visible = showBox;
      refs.boxEdges.scale.set(config.sizeX, config.sizeY, config.sizeZ);
      refs.boxEdges.position.set(config.centerX, config.centerY, config.centerZ);
      refs.boxFaces.scale.set(config.sizeX, config.sizeY, config.sizeZ);
      refs.boxFaces.position.set(config.centerX, config.centerY, config.centerZ);
    }
    // rebuild the binding-site census from the receptor atoms
    refs.bindingPoints.geometry.dispose();
    const replacement = buildBindingCloud(receptorAtoms, refs.bindingPoints.material, config);
    refs.bindingPoints.geometry = replacement.geometry;
    refs.bindingPoints.position.set(0, 0, 0);
    const insideNow = countAtomsInBox(receptorAtoms, config);
    refs.bindingPoints.visible = showHighlight && insideNow > 0;
  }, [config, ready, showBox, showHighlight, receptorAtoms]);

  // the receptor cloud — rebuilt when the atoms identity changes, with
  // the camera fitted to the combined structure (protein ∪ box ∪ ligand)
  useEffect(() => {
    const refs = sceneRef.current;
    if (refs === null || !ready) {
      return;
    }
    refs.receptorPoints.geometry.dispose();
    const replacement = buildCloud(receptorAtoms, refs.receptorPoints.material);
    refs.receptorPoints.geometry = replacement.geometry;
    refs.receptorPoints.visible = showProtein && receptorAtoms.length > 0;

    // the reference-frame triad: drawn at the receptor bbox minimum
    // corner in the receptor's OWN frame (never transformed)
    while (refs.axesGroup.children.length > 0) {
      const child = refs.axesGroup.children[refs.axesGroup.children.length - 1]!;
      refs.axesGroup.remove(child);
      if (child instanceof THREE.Line) {
        child.geometry.dispose();
        (child.material as THREE.Material).dispose();
      }
    }
    const fitBounds: { min: THREE.Vector3; max: THREE.Vector3 } | null =
      receptorAtoms.length > 0
        ? (() => {
            const min = new THREE.Vector3(Infinity, Infinity, Infinity);
            const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
            for (const a of receptorAtoms) {
              min.set(Math.min(min.x, a.x), Math.min(min.y, a.y), Math.min(min.z, a.z));
              max.set(Math.max(max.x, a.x), Math.max(max.y, a.y), Math.max(max.z, a.z));
            }
            return { min, max };
          })()
        : null;
    if (fitBounds !== null) {
      const span = Math.max(
        fitBounds.max.x - fitBounds.min.x,
        fitBounds.max.y - fitBounds.min.y,
        fitBounds.max.z - fitBounds.min.z,
        8,
      );
      const axisLength = span * 0.22;
      const origin = fitBounds.min.clone();
      const axes: readonly { dir: THREE.Vector3; color: number }[] = [
        { dir: new THREE.Vector3(1, 0, 0), color: AXIS_X_COLOR },
        { dir: new THREE.Vector3(0, 1, 0), color: AXIS_Y_COLOR },
        { dir: new THREE.Vector3(0, 0, 1), color: AXIS_Z_COLOR },
      ];
      for (const axis of axes) {
        const geometry = new THREE.BufferGeometry().setFromPoints([
          origin.clone(),
          origin.clone().addScaledVector(axis.dir, axisLength),
        ]);
        const line = new THREE.Line(
          geometry,
          new THREE.LineBasicMaterial({ color: axis.color }),
        );
        refs.axesGroup.add(line);
      }
    }

    // camera fit over the combined structure (protein ∪ box ∪ ligand)
    const centers: THREE.Vector3[] = [];
    let radius = 12;
    if (fitBounds !== null) {
      const size = fitBounds.max.clone().sub(fitBounds.min);
      const center = fitBounds.min.clone().addScaledVector(size, 0.5);
      centers.push(center);
      radius = Math.max(radius, size.length() / 2);
    }
    if (extents !== null) {
      centers.push(
        new THREE.Vector3(config.centerX, config.centerY, config.centerZ),
      );
      radius = Math.max(
        radius,
        Math.max(config.sizeX, config.sizeY, config.sizeZ) / 2,
      );
    }
    if (ligand !== null && ligand.atoms.length > 0) {
      const min = new THREE.Vector3(Infinity, Infinity, Infinity);
      const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
      for (const a of ligand.atoms) {
        min.set(Math.min(min.x, a.x), Math.min(min.y, a.y), Math.min(min.z, a.z));
        max.set(Math.max(max.x, a.x), Math.max(max.y, a.y), Math.max(max.z, a.z));
      }
      const size = max.clone().sub(min);
      centers.push(min.clone().addScaledVector(size, 0.5));
      radius = Math.max(radius, size.length() / 2);
    }
    if (centers.length > 0) {
      const group = new THREE.Vector3();
      for (const c of centers) {
        group.add(c);
      }
      group.divideScalar(centers.length);
      const distance = (radius / Math.tan((refs.camera.fov * Math.PI) / 360)) * 1.35 + 1;
      const direction = new THREE.Vector3(0.6, 0.45, 0.8).normalize();
      refs.camera.position.copy(group).addScaledVector(direction, distance);
      refs.controls.target.copy(group);
      refs.controls.minDistance = Math.max(radius * 0.25, 0.5);
      refs.controls.maxDistance = distance * 4;
      refs.controls.update();
    }
  }, [receptorAtoms, ready, showProtein, extents, config, hasLigand, ligand]);

  // the ligand group — ball-and-stick over the bonds the page parsed
  useEffect(() => {
    const refs = sceneRef.current;
    if (refs === null || !ready) {
      return;
    }
    while (refs.ligandGroup.children.length > 0) {
      const child = refs.ligandGroup.children[refs.ligandGroup.children.length - 1]!;
      refs.ligandGroup.remove(child);
      if (child instanceof THREE.Mesh) {
        child.material.dispose();
      }
    }
    if (ligand === null) {
      return;
    }
    for (const atom of ligand.atoms) {
      const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(cpkColor(atom.element)),
        metalness: 0.1,
        roughness: 0.45,
      });
      const mesh = new THREE.Mesh(refs.sphereGeometry, material);
      mesh.scale.setScalar(displayRadius(atom.element));
      mesh.position.set(atom.x, atom.y, atom.z);
      refs.ligandGroup.add(mesh);
    }
    for (const bond of ligand.bonds) {
      const a = ligand.atoms[bond.a];
      const b = ligand.atoms[bond.b];
      if (a === undefined || b === undefined) {
        continue;
      }
      const start = new THREE.Vector3(a.x, a.y, a.z);
      const end = new THREE.Vector3(b.x, b.y, b.z);
      const direction = end.clone().sub(start);
      const length = direction.length();
      if (length <= 1e-6) {
        continue;
      }
      const cylinder = new THREE.Mesh(refs.bondGeometry, refs.bondMaterial);
      cylinder.position.copy(start).add(end).multiplyScalar(0.5);
      cylinder.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        direction.clone().normalize(),
      );
      cylinder.scale.set(1, length, 1);
      refs.ligandGroup.add(cylinder);
    }
  }, [ligand, ready]);

  // the visibility toggles (Phase 9's required controls)
  useEffect(() => {
    const refs = sceneRef.current;
    if (refs === null || !ready) {
      return;
    }
    refs.receptorPoints.visible = showProtein && receptorAtoms.length > 0;
    const malformed = extents === null;
    refs.boxEdges.visible = showBox && !malformed;
    refs.boxFaces.visible = showBox && !malformed;
    refs.ligandGroup.visible = showLigand && hasLigand;
    refs.bindingPoints.visible =
      showHighlight && countAtomsInBox(receptorAtoms, config) > 0;
  }, [showProtein, showBox, showLigand, showHighlight, ready, receptorAtoms, config, extents, hasLigand]);

  const warningChips = warnings.map((warning) => (
    <span
      key={warning.code}
      role="status"
      className={
        warning.severity === "error"
          ? "rounded-full border border-[var(--color-destructive)] bg-[var(--color-destructive)]/10 px-2 py-0.5 text-xs text-[var(--color-destructive)]"
          : "rounded-full border border-[#e6b422] bg-[#e6b422]/10 px-2 py-0.5 text-xs text-[#a97d12]"
      }
    >
      {warning.message}
    </span>
  ));

  if (!hasReceptor) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Grid box 3D view</CardTitle>
          <CardDescription>
            The synchronized protein + box + ligand visualization (Phase 9).
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3 text-sm">
          <div
            role="status"
            className="flex items-center justify-center rounded-md border border-dashed border-[var(--color-border)] px-4 py-10 text-center text-[var(--color-muted-foreground)]"
          >
            <div>
              <p className="font-medium">No receptor atoms arrived yet.</p>
              <p className="mt-1 text-xs">
                Paste or prepare a receptor in the stage above — the 3D view
                renders only structures that actually arrived, never an
                imagined protein.
              </p>
            </div>
          </div>
          <p className="text-xs text-[var(--color-muted-foreground)]">
            box center ({formatCoord(config.centerX)}, {formatCoord(config.centerY)},{" "}
            {formatCoord(config.centerZ)}) Å · size {formatCoord(config.sizeX)} ×{" "}
            {formatCoord(config.sizeY)} × {formatCoord(config.sizeZ)} Å
          </p>
          <p className="text-xs text-[var(--color-muted-foreground)]">{REFERENCE_FRAME_NOTE}</p>
          <p className="text-xs text-[var(--color-muted-foreground)]">{SPACING_NOTE}</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Grid box 3D view</CardTitle>
        <CardDescription>
          The synchronized protein + box + ligand visualization (Phase 9) —
          orbit / zoom / pan; the numeric inputs stay the one mutation
          surface.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
          <label className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={showProtein}
              onChange={(event) => setShowProtein(event.target.checked)}
            />
            Protein
          </label>
          <label className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={showBox}
              onChange={(event) => setShowBox(event.target.checked)}
            />
            Grid box
          </label>
          <label className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={showLigand}
              disabled={!hasLigand}
              onChange={(event) => setShowLigand(event.target.checked)}
            />
            Ligand{hasLigand ? "" : " (none arrived)"}
          </label>
          <label className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={showHighlight}
              onChange={(event) => setShowHighlight(event.target.checked)}
            />
            Binding-site highlight
          </label>
        </div>
        {webgl === "unavailable" ? (
          <div
            role="status"
            className="flex items-center justify-center rounded-md border border-dashed border-[var(--color-border)] px-4 py-8 text-center text-sm text-[var(--color-muted-foreground)]"
            style={{ height: 380 }}
          >
            <div>
              <p className="font-medium">WebGL unavailable — the 3D view is disabled.</p>
              <p className="mt-1 text-xs">
                The numeric grid-box lane remains the scientific authority;
                this fallback is stated rather than a blank canvas.
              </p>
            </div>
          </div>
        ) : sceneError !== null ? (
          <div
            role="alert"
            className="flex items-center justify-center rounded-md border border-dashed border-[var(--color-border)] px-4 py-8 text-center text-sm text-[var(--color-destructive)]"
            style={{ height: 380 }}
          >
            <div>
              <p className="font-medium">The grid-box scene failed to construct.</p>
              <p className="mt-1 text-xs">{sceneError}</p>
            </div>
          </div>
        ) : (
          <div
            ref={containerRef}
            className="relative overflow-hidden rounded-md border border-[var(--color-border)] bg-[var(--color-muted)]"
            style={{ height: 380 }}
          />
        )}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-[var(--color-muted-foreground)]">
          <span>
            center ({formatCoord(config.centerX)}, {formatCoord(config.centerY)},{" "}
            {formatCoord(config.centerZ)}) Å
          </span>
          <span>
            size {formatCoord(config.sizeX)} × {formatCoord(config.sizeY)} ×{" "}
            {formatCoord(config.sizeZ)} Å
          </span>
          <span>volume {volume.toLocaleString("en-US")} Å³</span>
          <span>
            receptor atoms in box: {receptorInBox} / {receptorAtoms.length}
          </span>
          {ligandInBox !== null && ligand !== null ? (
            <span>
              ligand atoms in box: {ligandInBox} / {ligand.atoms.length}
            </span>
          ) : null}
          <span>{UNIT_NOTE}</span>
        </div>
        {warningChips.length > 0 ? (
          <div className="flex flex-wrap items-center gap-2">{warningChips}</div>
        ) : null}
        {receptorDiagnostics !== null ? (
          <p className="text-xs text-[var(--color-muted-foreground)]">
            receptor parse: {receptorDiagnostics}
          </p>
        ) : null}
        {ligand !== null && ligand.diagnostics !== null ? (
          <p className="text-xs text-[var(--color-muted-foreground)]">
            ligand parse: {ligand.diagnostics}
          </p>
        ) : null}
        <p className="text-xs text-[var(--color-muted-foreground)]">{REFERENCE_FRAME_NOTE}</p>
        <p className="text-xs text-[var(--color-muted-foreground)]">{SPACING_NOTE}</p>
      </CardContent>
    </Card>
  );
}
