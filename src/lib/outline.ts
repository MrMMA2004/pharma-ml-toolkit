/**
 * Lightweight symbol extraction for the Code Explorer outline panel.
 *
 * Pure, dependency-free heuristics per language — not a real parser, but
 * tuned for the pharmaml toolkit's Python / Markdown / TOML / YAML / shell
 * sources. Lines inside triple-quoted Python strings are skipped so
 * docstring prose never produces phantom symbols.
 */

export type OutlineKind =
  | "class"
  | "function"
  | "method"
  | "heading"
  | "section";

export interface OutlineSymbol {
  /** 1-based line number in the source file. */
  line: number;
  /** Symbol name (identifier or heading text, trimmed). */
  name: string;
  kind: OutlineKind;
  /** Nesting depth for indentation in the outline list (0 = top level). */
  depth: number;
}

/** Hard cap — very large generated files stay navigable. */
export const OUTLINE_MAX_SYMBOLS = 140;

/** Max characters kept for a symbol / heading name. */
const MAX_NAME = 72;

function cleanName(raw: string): string {
  const name = raw.trim().replace(/[`*_]+/g, "").trim();
  return name.length > MAX_NAME ? `${name.slice(0, MAX_NAME - 1)}…` : name;
}

/* ------------------------------- Python ------------------------------- */

const PY_SYMBOL_RE = /^(\s*)(?:async\s+)?(class|def)\s+([A-Za-z_]\w*)/;

/**
 * Advance the triple-quote string state across one source line.
 * Returns the code-visible part of the line (text outside strings) and the
 * string state to carry into the next line.
 */
function advanceStringState(
  line: string,
  inString: string | null
): { codePart: string; inString: string | null } {
  if (inString) {
    const close = line.indexOf(inString);
    if (close === -1) return { codePart: "", inString };
    // Delimiter closes here — whatever follows is code again.
    return advanceStringState(line.slice(close + 3), null);
  }
  const dq = line.indexOf('"""');
  const sq = line.indexOf("'''");
  let delim: string | null = null;
  let at = -1;
  if (dq !== -1 && (sq === -1 || dq < sq)) {
    delim = '"""';
    at = dq;
  } else if (sq !== -1) {
    delim = "'''";
    at = sq;
  }
  if (!delim) return { codePart: line, inString: null };
  const rest = line.slice(at + 3);
  const close = rest.indexOf(delim);
  if (close !== -1) {
    // Opened and closed on the same line (one-line docstring / expression).
    return advanceStringState(rest.slice(close + 3), null);
  }
  return { codePart: line.slice(0, at), inString: delim };
}

function pythonOutline(code: string): OutlineSymbol[] {
  const symbols: OutlineSymbol[] = [];
  let inString: string | null = null;
  const lines = code.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i] ?? "";
    const { codePart, inString: next } = advanceStringState(line, inString);
    inString = next;
    const match = PY_SYMBOL_RE.exec(codePart);
    if (!match) continue;
    const indent = match[1]?.length ?? 0;
    const isClass = match[2] === "class";
    const depth = Math.min(Math.floor(indent / 4), 2);
    symbols.push({
      line: i + 1,
      name: match[3] ?? "",
      kind: isClass ? "class" : indent === 0 ? "function" : "method",
      depth,
    });
  }
  return symbols;
}

/* ------------------------------ Markdown ------------------------------ */

const MD_HEADING_RE = /^(#{1,5})\s+(.+?)\s*#*\s*$/;
const MD_FRONTMATTER = /^---\s*$/;

function markdownOutline(code: string): OutlineSymbol[] {
  const symbols: OutlineSymbol[] = [];
  const lines = code.split("\n");
  let inFrontmatter = false;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i] ?? "";
    if (i === 0 && MD_FRONTMATTER.test(line)) {
      inFrontmatter = true;
      continue;
    }
    if (inFrontmatter) {
      if (MD_FRONTMATTER.test(line)) inFrontmatter = false;
      continue;
    }
    const match = MD_HEADING_RE.exec(line);
    if (!match) continue;
    const level = match[1]?.length ?? 1;
    const name = cleanName(match[2] ?? "");
    if (!name) continue;
    symbols.push({
      line: i + 1,
      name,
      kind: "heading",
      depth: Math.min(level - 1, 3),
    });
  }
  return symbols;
}

/* ----------------------------- TOML / INI ----------------------------- */

const SECTION_RE = /^\[+\s*([^\]]+?)\s*\]+/;

function tomlOutline(code: string): OutlineSymbol[] {
  const symbols: OutlineSymbol[] = [];
  const lines = code.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = (lines[i] ?? "").trim();
    if (line.startsWith("#")) continue;
    const match = SECTION_RE.exec(line);
    if (!match) continue;
    const name = match[1] ?? "";
    if (!name) continue;
    const dotted = name.split(".");
    symbols.push({
      line: i + 1,
      name: cleanName(dotted[dotted.length - 1] ?? name),
      kind: "section",
      depth: Math.min(dotted.length - 1, 2),
    });
  }
  return symbols;
}

/* -------------------------------- YAML -------------------------------- */

const YAML_KEY_RE = /^(\s*)([A-Za-z_][\w.-]*)\s*:(?:\s|$)/;

function yamlOutline(code: string): OutlineSymbol[] {
  const symbols: OutlineSymbol[] = [];
  const lines = code.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i] ?? "";
    const stripped = line.trimStart();
    if (stripped.startsWith("#") || stripped.startsWith("-")) continue;
    const match = YAML_KEY_RE.exec(line);
    if (!match) continue;
    const indent = match[1]?.length ?? 0;
    const depth = Math.min(Math.floor(indent / 2), 2);
    if (indent > 4) continue; // only top two levels — deeper keys are noise
    symbols.push({
      line: i + 1,
      name: match[2] ?? "",
      kind: "section",
      depth,
    });
  }
  return symbols;
}

/* ------------------------------ Shell -------------------------------- */

const SHELL_FN_RE = /^([A-Za-z_]\w*)\s*\(\s*\)\s*\{/;

function shellOutline(code: string): OutlineSymbol[] {
  const symbols: OutlineSymbol[] = [];
  const lines = code.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const match = SHELL_FN_RE.exec(lines[i] ?? "");
    if (!match) continue;
    symbols.push({
      line: i + 1,
      name: match[1] ?? "",
      kind: "function",
      depth: 0,
    });
  }
  return symbols;
}

/* ------------------------------ Dispatch ------------------------------ */

const PARSERS: Record<string, (code: string) => OutlineSymbol[]> = {
  python: pythonOutline,
  markdown: markdownOutline,
  toml: tomlOutline,
  ini: tomlOutline,
  yaml: yamlOutline,
  bash: shellOutline,
};

/**
 * Extract outline symbols for a file. Languages without a heuristic parser
 * (JSON, Dockerfile, plain text) return an empty array — callers hide the
 * outline panel in that case.
 */
export function extractOutline(code: string, language: string): OutlineSymbol[] {
  const parser = PARSERS[language];
  if (!parser) return [];
  return parser(code).slice(0, OUTLINE_MAX_SYMBOLS);
}

/** Human label for a symbol kind (tooltips + aria descriptions). */
export const OUTLINE_KIND_LABEL: Record<OutlineKind, string> = {
  class: "class",
  function: "function",
  method: "method",
  heading: "heading",
  section: "section",
};
