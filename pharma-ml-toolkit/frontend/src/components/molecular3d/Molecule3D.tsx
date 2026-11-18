/**
 * The part11 Phase Eighteen Three.js viewer — genuine 3D scientific
 * molecular visualization.
 *
 * Phase Eighteen's contract, enforced structurally here:
 *
 * * interactive rotation, zoom, pan (OrbitControls with damping);
 * * material + rendering configuration (MeshStandardMaterial per
 *   element, the CPK palette, antialiased renderer);
 * * clear molecular representation (ball-and-stick);
 * * atom/bond visualization (bond cylinders, display-only — the
 *   distance heuristic is stated in the required note prop);
 * * selection and highlighting (raycast pick → emissive highlight +
 *   the metadata callback);
 * * structure metadata (the counts + bounding box + element
 *   histogram strip);
 * * proper loading/error states (the WebGL-unavailable fallback is
 *   an honest rendered card, never a blank canvas);
 * * memory cleanup (every geometry/material/renderer disposed on
 *   unmount — the whole scene graph is rebuilt per data identity);
 * * resource-aware rendering (devicePixelRatio capped at 2, the RAF
 *   loop pauses offscreen via IntersectionObserver and on
 *   document.hidden);
 * * Three.js ONLY for genuine 3D scenes — 2D lanes stay Plotly.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  cpkColor,
  displayRadius,
  type InferredBond,
} from "@/services/molecular3d/pdbqt";

/** One rendered atom — only what actually arrived. */
export interface Molecule3DAtom {
  readonly element: string;
  readonly x: number;
  readonly y: number;
  readonly z: number;
  /** The metadata line for this atom (name/residue…), or null. */
  readonly label: string | null;
}

export interface Molecule3DSelection {
  readonly index: number;
  readonly element: string;
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly label: string | null;
}

export interface Molecule3DProps {
  /** The parsed atoms (the PDBQT parser's own output shape). */
  readonly atoms: readonly Molecule3DAtom[];
  /** The display bonds (the distance heuristic, display only). */
  readonly bonds: readonly InferredBond[];
  /**
   * The trajectory frames — `frames[i][atomIndex] = [x, y, z]`. When
   * present the viewer adds the frame playback lane (slider + play);
   * the atom positions render ONLY from the frames that arrived.
   */
  readonly frames?: readonly (readonly (readonly [number, number, number])[])[] | null;
  /** The honest display-heuristic note (required, never empty). */
  readonly bondNote: string;
  readonly height?: number;
  readonly title: string;
  readonly description?: string;
  readonly onAtomSelect?: (selection: Molecule3DSelection | null) => void;
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

interface AtomMesh extends THREE.Mesh<THREE.SphereGeometry, THREE.MeshStandardMaterial> {
  userData: { atomIndex: number };
}

interface SceneRefs {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  atomMeshes: AtomMesh[];
  bondGroup: THREE.Group;
  sphereGeometry: THREE.SphereGeometry;
  bondGeometry: THREE.CylinderGeometry;
  bondMaterial: THREE.MeshStandardMaterial;
  highlightMesh: THREE.Mesh<THREE.SphereGeometry, THREE.MeshBasicMaterial>;
  raf: number;
  resizeObserver: ResizeObserver;
  intersectionObserver: IntersectionObserver | null;
  visible: boolean;
}

const BOND_COLOR = 0x9aa0a6;
const HIGHLIGHT_COLOR = 0xe6b422;

/** (Re)position every atom mesh from the given positions source. */
function applyPositions(
  scene: SceneRefs,
  atoms: readonly Molecule3DAtom[],
  positions: readonly (readonly [number, number, number])[] | null,
): void {
  for (let i = 0; i < scene.atomMeshes.length; i += 1) {
    const mesh = scene.atomMeshes[i]!;
    const atom = atoms[i];
    const position = positions?.[i];
    if (position !== undefined) {
      mesh.position.set(position[0], position[1], position[2]);
    } else if (atom !== undefined) {
      mesh.position.set(atom.x, atom.y, atom.z);
    }
  }
}

/** Rebuild the bond cylinders for the current positions (display only). */
function rebuildBonds(
  scene: SceneRefs,
  bondList: readonly InferredBond[],
): void {
  while (scene.bondGroup.children.length > 0) {
    const child = scene.bondGroup.children[scene.bondGroup.children.length - 1]!;
    scene.bondGroup.remove(child);
  }
  for (const bond of bondList) {
    const a = scene.atomMeshes[bond.a];
    const b = scene.atomMeshes[bond.b];
    if (a === undefined || b === undefined) {
      continue;
    }
    const start = a.position;
    const end = b.position;
    const direction = end.clone().sub(start);
    const length = direction.length();
    if (length <= 1e-6) {
      continue;
    }
    const cylinder = new THREE.Mesh(scene.bondGeometry, scene.bondMaterial);
    cylinder.position.copy(start).add(end).multiplyScalar(0.5);
    cylinder.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      direction.clone().normalize(),
    );
    cylinder.scale.set(1, length, 1);
    scene.bondGroup.add(cylinder);
  }
}

/** The complete teardown — every resource disposed (memory cleanup). */
function disposeScene(scene: SceneRefs): void {
  cancelAnimationFrame(scene.raf);
  scene.intersectionObserver?.disconnect();
  scene.resizeObserver.disconnect();
  scene.controls.dispose();
  for (const mesh of scene.atomMeshes) {
    scene.scene.remove(mesh);
    mesh.material.dispose();
  }
  scene.scene.remove(scene.bondGroup);
  scene.scene.remove(scene.highlightMesh);
  (scene.highlightMesh.geometry as THREE.SphereGeometry).dispose();
  (scene.highlightMesh.material as THREE.MeshBasicMaterial).dispose();
  scene.sphereGeometry.dispose();
  scene.bondGeometry.dispose();
  scene.bondMaterial.dispose();
  scene.renderer.dispose();
  scene.renderer.domElement.remove();
}

export function Molecule3D({
  atoms,
  bonds,
  frames,
  bondNote,
  height = 360,
  title,
  description,
  onAtomSelect,
}: Molecule3DProps) {
  if (bondNote.trim() === "") {
    throw new Error("Molecule3D: the display-bond honesty note is required");
  }
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<SceneRefs | null>(null);
  const frameIndexRef = useRef(0);
  const [webgl] = useState<WebglSupport>(() => probeWebgl());
  const [ready, setReady] = useState(false);
  const [sceneError, setSceneError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Molecule3DSelection | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);

  const frameCount = frames !== null && frames !== undefined ? frames.length : 0;

  const elementCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const atom of atoms) {
      counts.set(atom.element, (counts.get(atom.element) ?? 0) + 1);
    }
    return [...counts.entries()]
      .map(([element, count]) => ({ element, count }))
      .sort((a, b) => b.count - a.count || a.element.localeCompare(b.element));
  }, [atoms]);

  const bounding = useMemo(() => {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (const a of atoms) {
      minX = Math.min(minX, a.x); maxX = Math.max(maxX, a.x);
      minY = Math.min(minY, a.y); maxY = Math.max(maxY, a.y);
      minZ = Math.min(minZ, a.z); maxZ = Math.max(maxZ, a.z);
    }
    if (atoms.length === 0) {
      return null;
    }
    const sizeX = maxX - minX;
    const sizeY = maxY - minY;
    const sizeZ = maxZ - minZ;
    const center = new THREE.Vector3(
      (minX + maxX) / 2,
      (minY + maxY) / 2,
      (minZ + maxZ) / 2,
    );
    const radius = Math.max(sizeX, sizeY, sizeZ, 1e-6) / 2;
    return { center, radius, sizeX, sizeY, sizeZ };
  }, [atoms]);

  const currentPositions = useMemo((): readonly (readonly [number, number, number])[] | null => {
    if (frames === null || frames === undefined) {
      return null;
    }
    const frame = frames[Math.min(frameIndex, frames.length - 1)];
    if (frame === undefined) {
      return null;
    }
    return frame;
  }, [frames, frameIndex]);

  // the scene lifecycle — built once per webgl availability, torn
  // down completely on unmount (every resource disposed)
  useEffect(() => {
    if (webgl !== "available" || containerRef.current === null) {
      return;
    }
    const container = containerRef.current;
    let scene: SceneRefs | null = null;
    let announcedReady = false;
    try {
      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      // resource-aware: the device pixel ratio capped at 2
      renderer.setPixelRatio(Math.min(window.devicePixelRatio ?? 1, 2));
      renderer.setSize(container.clientWidth || 640, height);
      container.appendChild(renderer.domElement);
      const camera = new THREE.PerspectiveCamera(45, (container.clientWidth || 640) / height, 0.1, 10000);
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      const sceneThree = new THREE.Scene();
      sceneThree.add(new THREE.AmbientLight(0xffffff, 0.55));
      const key = new THREE.DirectionalLight(0xffffff, 1.05);
      key.position.set(1, 1.6, 0.9);
      sceneThree.add(key);
      const fill = new THREE.DirectionalLight(0xffffff, 0.4);
      fill.position.set(-1, -0.4, -1);
      sceneThree.add(fill);
      const sphereGeometry = new THREE.SphereGeometry(1, 24, 16);
      const bondGeometry = new THREE.CylinderGeometry(0.09, 0.09, 1, 12, 1, true);
      const bondMaterial = new THREE.MeshStandardMaterial({
        color: BOND_COLOR,
        metalness: 0.15,
        roughness: 0.55,
      });
      const bondGroup = new THREE.Group();
      sceneThree.add(bondGroup);
      const highlightMesh = new THREE.Mesh(
        new THREE.SphereGeometry(1, 24, 16),
        new THREE.MeshBasicMaterial({
          color: HIGHLIGHT_COLOR,
          wireframe: true,
          transparent: true,
          opacity: 0.9,
        }),
      );
      highlightMesh.visible = false;
      sceneThree.add(highlightMesh);
      scene = {
        renderer,
        scene: sceneThree,
        camera,
        controls,
        atomMeshes: [],
        bondGroup,
        sphereGeometry,
        bondGeometry,
        bondMaterial,
        highlightMesh,
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
      scene.resizeObserver.observe(container);
      // resource-aware: pause the RAF loop offscreen
      const intersectionObserver = new IntersectionObserver((entries) => {
        if (scene === null) {
          return;
        }
        scene.visible = entries.some((e) => e.isIntersecting);
      });
      intersectionObserver.observe(container);
      scene.intersectionObserver = intersectionObserver;

      const renderLoop = () => {
        if (scene === null) {
          return;
        }
        scene.raf = requestAnimationFrame(renderLoop);
        if (!announcedReady) {
          // the readiness announcement rides the first rAF tick —
          // never a synchronous setState inside the effect body
          announcedReady = true;
          setReady(true);
          setSceneError(null);
        }
        if (!scene.visible || document.hidden) {
          return;
        }
        scene.controls.update();
        scene.renderer.render(scene.scene, scene.camera);
      };
      scene.raf = requestAnimationFrame(renderLoop);
      sceneRef.current = scene;
    } catch (error) {
      // a partial scene is torn down completely (never a leaked
      // half-built renderer), and the failure announcement rides a
      // microtask — never a synchronous setState in the effect body
      if (scene !== null) {
        disposeScene(scene);
      }
      sceneRef.current = null;
      const message =
        error instanceof Error
          ? error.message
          : "the 3D scene could not be constructed";
      queueMicrotask(() => {
        setSceneError(message);
        setReady(false);
      });
    }
    return () => {
      if (scene === null) {
        return;
      }
      disposeScene(scene);
      sceneRef.current = null;
      setReady(false);
    };
  }, [webgl, height]);

  // the atom meshes — rebuilt when the atom identity changes
  useEffect(() => {
    const scene = sceneRef.current;
    if (scene === null || !ready) {
      return;
    }
    for (const mesh of scene.atomMeshes) {
      scene.scene.remove(mesh);
      mesh.material.dispose();
    }
    scene.atomMeshes.length = 0;
    atoms.forEach((atom, index) => {
      const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(cpkColor(atom.element)),
        metalness: 0.1,
        roughness: 0.45,
      });
      const mesh = new THREE.Mesh(scene.sphereGeometry, material) as AtomMesh;
      const radius = displayRadius(atom.element);
      mesh.scale.setScalar(radius);
      mesh.position.set(atom.x, atom.y, atom.z);
      mesh.userData.atomIndex = index;
      scene.scene.add(mesh);
      scene.atomMeshes.push(mesh);
    });
    applyPositions(scene, atoms, currentPositions);
    rebuildBonds(scene, bonds);
    if (bounding !== null) {
      // camera fit: the bounding sphere of the structure that arrived
      const distance =
        (bounding.radius / Math.tan((scene.camera.fov * Math.PI) / 360)) * 1.35 + 1;
      const direction = new THREE.Vector3(0.6, 0.45, 0.8).normalize();
      scene.camera.position.copy(bounding.center).addScaledVector(direction, distance);
      scene.controls.target.copy(bounding.center);
      scene.controls.minDistance = Math.max(bounding.radius * 0.25, 0.5);
      scene.controls.maxDistance = distance * 4;
      scene.controls.update();
    }
  }, [atoms, bonds, ready, bounding, currentPositions]);

  // the frame playback — positions applied only from frames that arrived
  useEffect(() => {
    const scene = sceneRef.current;
    if (scene === null || !ready) {
      return;
    }
    applyPositions(scene, atoms, currentPositions);
    rebuildBonds(scene, bonds);
  }, [currentPositions, ready, atoms, bonds]);

  // the playback ticker
  useEffect(() => {
    if (!playing || frameCount < 2) {
      return;
    }
    const timer = window.setInterval(() => {
      frameIndexRef.current = (frameIndexRef.current + 1) % frameCount;
      setFrameIndex(frameIndexRef.current);
    }, 220);
    return () => window.clearInterval(timer);
  }, [playing, frameCount]);

  // the selection lane — raycast pick with the drag threshold
  useEffect(() => {
    const scene = sceneRef.current;
    if (scene === null || !ready) {
      return;
    }
    let downX = 0;
    let downY = 0;
    const onPointerDown = (event: PointerEvent) => {
      downX = event.clientX;
      downY = event.clientY;
    };
    const onPointerUp = (event: PointerEvent) => {
      if (Math.abs(event.clientX - downX) + Math.abs(event.clientY - downY) > 6) {
        return; // an orbit drag, never a pick
      }
      const rect = scene.renderer.domElement.getBoundingClientRect();
      const ndc = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      );
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(ndc, scene.camera);
      const hits = raycaster.intersectObjects(scene.atomMeshes, false);
      if (hits.length === 0) {
        scene.highlightMesh.visible = false;
        setSelected(null);
        onAtomSelect?.(null);
        return;
      }
      const mesh = hits[0]!.object as AtomMesh;
      const index = mesh.userData.atomIndex;
      const atom = atoms[index];
      if (atom === undefined) {
        return;
      }
      scene.highlightMesh.position.copy(mesh.position);
      scene.highlightMesh.scale.setScalar(displayRadius(atom.element) * 1.7);
      scene.highlightMesh.visible = true;
      const positions = currentPositions;
      const selection: Molecule3DSelection = {
        index,
        element: atom.element,
        x: positions?.[index]?.[0] ?? mesh.position.x,
        y: positions?.[index]?.[1] ?? mesh.position.y,
        z: positions?.[index]?.[2] ?? mesh.position.z,
        label: atom.label,
      };
      setSelected(selection);
      onAtomSelect?.(selection);
    };
    scene.renderer.domElement.addEventListener("pointerdown", onPointerDown);
    scene.renderer.domElement.addEventListener("pointerup", onPointerUp);
    return () => {
      scene.renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      scene.renderer.domElement.removeEventListener("pointerup", onPointerUp);
    };
  }, [atoms, ready, currentPositions, onAtomSelect]);

  if (atoms.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{title}</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-[var(--color-muted-foreground)]">
          <p>Nothing to render yet — the 3D view draws only structures that actually arrived.</p>
          {description ? <p className="mt-1">{description}</p> : null}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
        <p className="text-xs text-[var(--color-muted-foreground)]">
          {description ?? "Ball-and-stick, interactive orbit / zoom / pan — click an atom to inspect it."}
        </p>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {webgl === "unavailable" ? (
          <div
            role="status"
            className="flex items-center justify-center rounded-md border border-dashed border-[var(--color-border)] px-4 py-8 text-center text-sm text-[var(--color-muted-foreground)]"
            style={{ height }}
          >
            <div>
              <p className="font-medium">WebGL unavailable — the 3D view is disabled.</p>
              <p className="mt-1 text-xs">
                The textual and 2D lanes remain the scientific authority; this
                fallback is stated rather than a blank canvas (Phase
                Eighteen&apos;s WebView2/hardware limitation fallback).
              </p>
            </div>
          </div>
        ) : sceneError !== null ? (
          <div
            role="alert"
            className="flex items-center justify-center rounded-md border border-dashed border-[var(--color-border)] px-4 py-8 text-center text-sm text-[var(--color-destructive)]"
            style={{ height }}
          >
            <div>
              <p className="font-medium">The 3D scene failed to construct.</p>
              <p className="mt-1 text-xs">{sceneError}</p>
            </div>
          </div>
        ) : (
          <div
            ref={containerRef}
            className="relative overflow-hidden rounded-md border border-[var(--color-border)] bg-[var(--color-muted)]"
            style={{ height }}
          />
        )}
        {frameCount > 0 ? (
          <div className="flex flex-wrap items-center gap-3 text-xs">
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={frameCount < 2}
              onClick={() => setPlaying((p) => !p)}
            >
              {playing ? "Pause" : "Play"}
            </Button>
            <label className="flex items-center gap-2">
              <span className="text-[var(--color-muted-foreground)]">
                Frame {frameIndex + 1} / {frameCount}
              </span>
              <input
                type="range"
                min={0}
                max={frameCount - 1}
                value={Math.min(frameIndex, frameCount - 1)}
                onChange={(event) => {
                  const value = Number.parseInt(event.target.value, 10);
                  frameIndexRef.current = value;
                  setFrameIndex(value);
                  setPlaying(false);
                }}
                className="w-48"
                aria-label="Trajectory frame"
              />
            </label>
          </div>
        ) : null}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-[var(--color-muted-foreground)]">
          <span>
            {atoms.length} atoms · {bonds.length} display bonds ·{" "}
            {elementCounts
              .slice(0, 6)
              .map((e) => `${e.element} ×${e.count}`)
              .join(" · ")}
            {elementCounts.length > 6 ? " · …" : ""}
          </span>
          {bounding !== null ? (
            <span>
              bbox {bounding.sizeX.toFixed(1)} × {bounding.sizeY.toFixed(1)} ×{" "}
              {bounding.sizeZ.toFixed(1)} Å
            </span>
          ) : null}
          {selected !== null ? (
            <span className="font-medium text-[var(--color-foreground)]">
              selected atom #{selected.index + 1} — {selected.element} (
              {selected.x.toFixed(2)}, {selected.y.toFixed(2)}, {selected.z.toFixed(2)} Å)
              {selected.label ? ` · ${selected.label}` : ""}
            </span>
          ) : null}
        </div>
        <p className="text-xs text-[var(--color-muted-foreground)]">{bondNote}</p>
      </CardContent>
    </Card>
  );
}
