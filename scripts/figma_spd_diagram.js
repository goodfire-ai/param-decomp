// SPD Parameter Decomposition Diagram for Figma
// Usage: In Figma, go to Plugins > Development > New Plugin > "Run once"
// Paste this code into the plugin's code.ts (remove any default code first)

// ── Configuration ──────────────────────────────────────────────────────
const NUM_COMPONENTS = 4;
const CELL_SIZE = 14;
const CELL_GAP = 2;
const MATRIX_GAP = 28;
const COMPONENT_GAP = 60;
const LABEL_HEIGHT = 24;
const SECTION_LABEL_HEIGHT = 36;

// Caricature dimensions for a 1-layer transformer
const MATRICES = [
  { name: "W_E",       rows: 8, cols: 5, hueBase: 210 },  // (vocab, d_model)
  { name: "W_Q",       rows: 5, cols: 4, hueBase: 340 },  // (d_model, d_head)
  { name: "W_K",       rows: 5, cols: 4, hueBase: 20  },  // (d_model, d_head)
  { name: "W_V",       rows: 5, cols: 4, hueBase: 50  },  // (d_model, d_head)
  { name: "W_O",       rows: 4, cols: 5, hueBase: 120 },  // (d_head, d_model)
  { name: "W_mlp_in",  rows: 5, cols: 8, hueBase: 160 },  // (d_model, d_mlp)
  { name: "W_mlp_out", rows: 8, cols: 5, hueBase: 190 },  // (d_mlp, d_model)
  { name: "W_U",       rows: 5, cols: 8, hueBase: 270 },  // (d_model, vocab)
];

// ── Color helpers ──────────────────────────────────────────────────────

// Simple seeded PRNG for reproducibility
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// HSL to RGB (returns {r, g, b} in [0, 1])
function hslToRgb(h, s, l) {
  h = ((h % 360) + 360) % 360;
  s = Math.max(0, Math.min(1, s));
  l = Math.max(0, Math.min(1, l));
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let r1, g1, b1;
  if (h < 60)       { r1 = c; g1 = x; b1 = 0; }
  else if (h < 120) { r1 = x; g1 = c; b1 = 0; }
  else if (h < 180) { r1 = 0; g1 = c; b1 = x; }
  else if (h < 240) { r1 = 0; g1 = x; b1 = c; }
  else if (h < 300) { r1 = x; g1 = 0; b1 = c; }
  else              { r1 = c; g1 = 0; b1 = x; }
  return { r: r1 + m, g: g1 + m, b: b1 + m };
}

// Generate a stable color for each cell in a matrix
function generateMatrixColors(matrixDef, seed) {
  const rng = mulberry32(seed);
  const colors = [];
  for (let r = 0; r < matrixDef.rows; r++) {
    const row = [];
    for (let c = 0; c < matrixDef.cols; c++) {
      const hue = matrixDef.hueBase + rng() * 40 - 20;
      const sat = 0.5 + rng() * 0.4;
      const lit = 0.35 + rng() * 0.35;
      row.push(hslToRgb(hue, sat, lit));
    }
    colors.push(row);
  }
  return colors;
}

// Generate opacity splits: for each cell, N_COMPONENTS opacities that sum to 1
function generateOpacitySplits(rows, cols, numComponents, seed) {
  const rng = mulberry32(seed);
  // splits[comp][row][col] = opacity
  const splits = Array.from({ length: numComponents }, () =>
    Array.from({ length: rows }, () => Array(cols).fill(0))
  );
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      // Dirichlet-like: sample exponentials, normalize
      const raw = [];
      for (let k = 0; k < numComponents; k++) {
        raw.push(-Math.log(1 - rng() * 0.999));  // exponential(1)
      }
      const sum = raw.reduce((a, b) => a + b, 0);
      for (let k = 0; k < numComponents; k++) {
        splits[k][r][c] = raw[k] / sum;
      }
    }
  }
  return splits;
}

// ── Figma node builders ────────────────────────────────────────────────

function createText(text, x, y, fontSize, isBold) {
  const node = figma.createText();
  node.x = x;
  node.y = y;
  node.fontSize = fontSize || 14;
  node.fontName = { family: "Inter", style: isBold ? "Bold" : "Regular" };
  node.characters = text;
  node.fills = [{ type: "SOLID", color: { r: 0.15, g: 0.15, b: 0.15 } }];
  return node;
}

function createCell(x, y, color, opacity) {
  const rect = figma.createRectangle();
  rect.x = x;
  rect.y = y;
  rect.resize(CELL_SIZE, CELL_SIZE);
  rect.cornerRadius = 2;
  rect.fills = [{ type: "SOLID", color: color, opacity: opacity }];
  rect.strokes = [{ type: "SOLID", color: { r: 0.85, g: 0.85, b: 0.85 } }];
  rect.strokeWeight = 0.5;
  return rect;
}

function matrixPixelWidth(cols) {
  return cols * (CELL_SIZE + CELL_GAP) - CELL_GAP;
}

function matrixPixelHeight(rows) {
  return rows * (CELL_SIZE + CELL_GAP) - CELL_GAP;
}

// Build one matrix grid at (x, y) with given colors and per-cell opacity
function buildMatrixGrid(frame, matrixDef, colors, opacities, x, y) {
  for (let r = 0; r < matrixDef.rows; r++) {
    for (let c = 0; c < matrixDef.cols; c++) {
      const cx = x + c * (CELL_SIZE + CELL_GAP);
      const cy = y + r * (CELL_SIZE + CELL_GAP);
      const op = opacities ? opacities[r][c] : 1.0;
      const cell = createCell(cx, cy, colors[r][c], op);
      frame.appendChild(cell);
    }
  }
}

// ── Main layout ────────────────────────────────────────────────────────

async function main() {
  // Load fonts
  await figma.loadFontAsync({ family: "Inter", style: "Regular" });
  await figma.loadFontAsync({ family: "Inter", style: "Bold" });

  const page = figma.currentPage;

  // Pre-compute colors and opacity splits for each matrix
  const allColors = MATRICES.map((m, i) => generateMatrixColors(m, 42 + i * 100));
  const allSplits = MATRICES.map((m, i) =>
    generateOpacitySplits(m.rows, m.cols, NUM_COMPONENTS, 7777 + i * 137)
  );

  // Compute layout: how wide is one "model depiction" (all matrices in a row)
  const maxMatrixHeight = Math.max(...MATRICES.map(m => matrixPixelHeight(m.rows)));
  let totalMatricesWidth = 0;
  const matrixXOffsets = [];
  for (const m of MATRICES) {
    matrixXOffsets.push(totalMatricesWidth);
    totalMatricesWidth += matrixPixelWidth(m.cols) + MATRIX_GAP;
  }
  totalMatricesWidth -= MATRIX_GAP;  // no trailing gap

  const sectionY = 0;
  const gridTopY = sectionY + SECTION_LABEL_HEIGHT + LABEL_HEIGHT + 8;

  // ── Original model ──────────────────────────────────────────────────
  const origLabel = createText("Original Model", 0, sectionY, 20, true);
  page.appendChild(origLabel);

  for (let mi = 0; mi < MATRICES.length; mi++) {
    const m = MATRICES[mi];
    const mx = matrixXOffsets[mi];

    // Matrix name label
    const label = createText(m.name, mx, sectionY + SECTION_LABEL_HEIGHT, 11, false);
    label.fills = [{ type: "SOLID", color: { r: 0.4, g: 0.4, b: 0.4 } }];
    page.appendChild(label);

    // Full-opacity grid
    const fullOpacity = Array.from({ length: m.rows }, () => Array(m.cols).fill(1.0));
    buildMatrixGrid(page, m, allColors[mi], fullOpacity, mx, gridTopY);
  }

  // ── "=" sign ────────────────────────────────────────────────────────
  const eqX = totalMatricesWidth + 30;
  const eqY = gridTopY + maxMatrixHeight / 2 - 14;
  const eqSign = createText("=", eqX, eqY, 28, true);
  page.appendChild(eqSign);

  // ── Component copies ────────────────────────────────────────────────
  const componentsStartX = eqX + 40;

  for (let comp = 0; comp < NUM_COMPONENTS; comp++) {
    const compX = componentsStartX;
    const compY = sectionY + comp * (maxMatrixHeight + LABEL_HEIGHT + SECTION_LABEL_HEIGHT + COMPONENT_GAP);
    const compGridY = compY + SECTION_LABEL_HEIGHT + LABEL_HEIGHT + 8;

    // Component label
    const compLabel = createText(
      `Component ${comp + 1}  (α = random, Σα = 1)`,
      compX, compY, 16, true
    );
    compLabel.fills = [{ type: "SOLID", color: hslToRgb(200 + comp * 40, 0.6, 0.35) }];
    page.appendChild(compLabel);

    for (let mi = 0; mi < MATRICES.length; mi++) {
      const m = MATRICES[mi];
      const mx = compX + matrixXOffsets[mi];

      // Matrix name
      const label = createText(m.name, mx, compY + SECTION_LABEL_HEIGHT, 11, false);
      label.fills = [{ type: "SOLID", color: { r: 0.4, g: 0.4, b: 0.4 } }];
      page.appendChild(label);

      // Grid with component-specific opacity
      buildMatrixGrid(page, m, allColors[mi], allSplits[mi][comp], mx, compGridY);
    }

    // "+" sign between components (except after the last)
    if (comp < NUM_COMPONENTS - 1) {
      const plusY = compGridY + maxMatrixHeight + COMPONENT_GAP / 2 - 14;
      const plusX = compX + totalMatricesWidth / 2 - 8;
      const plus = createText("+", plusX, plusY, 24, true);
      plus.fills = [{ type: "SOLID", color: { r: 0.5, g: 0.5, b: 0.5 } }];
      page.appendChild(plus);
    }
  }

  // Zoom to fit
  figma.viewport.scrollAndZoomIntoView(page.children);
  figma.closePlugin("SPD diagram created!");
}

main();
