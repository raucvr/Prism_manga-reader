/**
 * Demo fixtures for verifying the job view without a backend.
 *
 *   /?demo=1      in-progress v3 job: 12 panels, 3 chapters, 2 pending, 1 failed. The words are typeset into the
 *                 pictures (so the text under them starts closed); panel 10 is a board panel with a bar chart from the
 *                 paper's Table 2; with the information layer (a "More detail" text on every panel but one, a paper
 *                 card and its coverage) and an abstract check score
 *   /?demo=strip  completed job in the two-panel "strip" layout, as an older job without the information layer
 *   /?demo=early  job still storyboarding (no panels yet: full progress card)
 *
 * Images are generated SVG data URIs. This module is loaded with a dynamic
 * import, so it is not part of the main bundle.
 */

import type { JobBoard, JobChapter, JobCoverage, JobPanel, JobPaperCard, JobState } from "@/lib/api";

type Speaker = "papi" | "kumo" | "nezu";

interface DemoPanelSpec {
  title: string;
  characters: Speaker[];
  dialogue: Array<[Speaker, string]>;
  narration: string;
  /** The precise facts behind the panel ("" = none). */
  detail: string;
  /** Key points of the paper the panel explains (see COVERAGE). */
  covers: string[];
}

const CHAPTERS: Array<{ id: string; title: string; panels: DemoPanelSpec[] }> = [
  {
    id: "hook",
    title: "Why care?",
    panels: [
      {
        title: "The Hook",
        characters: ["papi", "kumo"],
        dialogue: [
          ["papi", "Ever wondered how translation apps got so good, so fast?"],
          ["kumo", "Wait, they weren't always this good?"],
        ],
        narration: "Before 2017, the best language models read text one word at a time.",
        detail:
          "The paper proposes the Transformer, a sequence-to-sequence model built entirely on attention, with no recurrence and no convolutions. Before it, the best translation systems were recurrent or convolutional encoder–decoders, usually connected through an attention mechanism.",
        covers: ["P3"],
      },
      {
        title: "One Word at a Time",
        characters: ["kumo"],
        dialogue: [["kumo", "Old models read like this: one… word… at… a… time."]],
        narration: "These were recurrent neural networks (RNNs). Each step had to wait for the one before it.",
        detail:
          "A recurrent network computes a hidden state hₜ from the previous state hₜ₋₁ and the word at position t. This inherently sequential computation rules out parallelization within a training example, which becomes critical for longer sequences because memory limits batching across examples.",
        covers: ["P1"],
      },
      {
        title: "The Forgetting Problem",
        characters: ["nezu", "kumo"],
        dialogue: [
          ["nezu", "By the end of a long sentence, I forget how it started!"],
          ["kumo", "That's the long-range dependency problem."],
        ],
        narration: "Information from early words fades as it passes through many steps.",
        detail:
          "Learning long-range dependencies is a key challenge: the longer the path a signal travels between two words, the harder the link is to learn. In a recurrent layer that path grows with the distance, O(n) steps; a self-attention layer connects any two positions in O(1) steps.",
        covers: ["P2", "P11"],
      },
      {
        title: "Too Slow to Scale",
        characters: ["papi"],
        dialogue: [["papi", "And since every step waits for the last one, the GPUs mostly sit idle."]],
        narration: "Sequential computation makes training on huge datasets slow and expensive.",
        detail:
          "A recurrent layer needs O(n) sequential operations for a sentence of n words; a self-attention layer needs O(1), so all positions are computed in parallel. Self-attention is also cheaper per layer when n is smaller than the representation size d, which is usually the case for sentences in translation.",
        covers: ["P1", "P11"],
      },
    ],
  },
  {
    id: "idea",
    title: "The big idea: attention",
    panels: [
      {
        title: "What If We Just Look?",
        characters: ["papi", "nezu"],
        dialogue: [
          ["papi", "What if every word could look at every other word at once?"],
          ["nezu", "All at the same time?!"],
        ],
        narration: "This is self-attention: each word compares itself directly with all the others.",
        detail:
          "Self-attention relates every position of a sequence to every other position to compute a new representation of it. The Transformer uses attention in three ways: self-attention in the encoder, masked self-attention in the decoder (a word cannot look at later words) and encoder–decoder attention.",
        covers: ["P3"],
      },
      {
        title: "Queries, Keys and Values",
        characters: ["kumo"],
        dialogue: [["kumo", "Each word asks a question (query), shows a label (key) and carries content (value)."]],
        narration: "Attention scores come from matching each query against every key.",
        detail:
          "Queries, keys and values are learned linear projections of the word vectors. Each output is a weighted sum of the values, where the weight of each value comes from how well the query matches the corresponding key.",
        covers: ["P4"],
      },
      {
        title: "Scaled Dot-Product",
        characters: ["nezu", "kumo"],
        dialogue: [
          ["nezu", "Multiply, scale by √d, softmax… done!"],
          ["kumo", "The scaling keeps the softmax from getting too spiky."],
        ],
        narration: "Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V",
        detail:
          "The dot products of a query with all keys are divided by √dₖ (dₖ = 64) and passed through a softmax to get the weights on the values. Without the scaling, large dot products push the softmax into regions with extremely small gradients.",
        covers: ["P4"],
      },
      {
        title: "Many Heads Are Better",
        characters: ["papi", "kumo", "nezu"],
        dialogue: [["papi", "Why have one attention when you can have eight?"]],
        narration: "Multi-head attention lets the model follow grammar, meaning and position at the same time.",
        detail:
          "The model runs h = 8 attention heads in parallel, each on 64-dimensional projections, then concatenates and projects their outputs, at about the cost of one full-size head. In the ablation, single-head attention was 0.9 BLEU worse than the best setting, and too many heads also hurt quality.",
        covers: ["P5", "P12"],
      },
    ],
  },
  {
    id: "impact",
    title: "Why it changed everything",
    panels: [
      {
        title: "Remembering Order",
        characters: ["kumo"],
        dialogue: [["kumo", "No recurrence means no built-in order… so we add a wave pattern to each position!"]],
        narration: "Positional encodings use sine and cosine waves so the model knows where each word sits.",
        detail:
          "Each position gets a 512-dimensional encoding added to its word embedding: PE(pos, 2i) = sin(pos / 10000^(2i/512)), and cosine for the odd dimensions. Learned position embeddings gave nearly identical results; the sine version was kept because it may extrapolate to sequences longer than those seen in training.",
        covers: ["P6"],
      },
      {
        title: "Faster and Better",
        characters: ["papi", "nezu"],
        dialogue: [
          ["papi", "It trained in 3.5 days on 8 GPUs…"],
          ["nezu", "…and beat every previous translation score!"],
        ],
        narration: "The Transformer set a new state of the art on English→German translation (28.4 BLEU).",
        detail:
          "On WMT 2014 English→German (about 4.5 million sentence pairs) the big Transformer reaches 28.4 BLEU, more than 2.0 BLEU above the best previous models, ensembles included. It trained for 3.5 days on 8 NVIDIA P100 GPUs; the base model reaches 27.3 BLEU after 12 hours.",
        covers: ["P8", "P9", "P15"],
      },
      {
        title: "The Family Tree",
        characters: ["nezu", "kumo"],
        dialogue: [
          ["nezu", "Wait, so BERT and GPT come from this?"],
          ["kumo", "Every single one of them!"],
        ],
        narration: "Almost every modern language model is built from Transformer blocks.",
        // Not about the paper itself: no detail and no key point (the "More detail" section stays hidden).
        detail: "",
        covers: [],
      },
      {
        title: "Your Turn",
        characters: ["papi", "kumo", "nezu"],
        dialogue: [["papi", "Next time an app finishes your sentence, say thanks to attention!"]],
        narration: "Paper: Vaswani et al., “Attention Is All You Need”, NeurIPS 2017.",
        detail:
          "The authors conclude that the Transformer, the first sequence transduction model based entirely on attention, trains significantly faster than recurrent or convolutional models for translation. Next, they plan to apply it to images, audio and video with local, restricted attention, and to make generation less sequential.",
        covers: ["P3", "P14"],
      },
    ],
  },
];

// ---------------------------------------------------------------------------
// Placeholder artwork (SVG)
// ---------------------------------------------------------------------------

const PANEL_W = 1024;
const PANEL_H = 1536;

const CHARACTER_STYLE: Record<Speaker, { body: string; accent: string }> = {
  papi: { body: "#fbd5db", accent: "#eb6079" },
  kumo: { body: "#ffffff", accent: "#36bbfa" },
  nezu: { body: "#e8e0f0", accent: "#8b74b8" },
};

function escapeXml(text: string): string {
  return text.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

function face(cx: number, cy: number, r: number): string {
  const eye = r * 0.07;
  return [
    `<circle cx="${cx - r * 0.3}" cy="${cy - r * 0.05}" r="${eye}" fill="#3b3b3b"/>`,
    `<circle cx="${cx + r * 0.3}" cy="${cy - r * 0.05}" r="${eye}" fill="#3b3b3b"/>`,
    `<ellipse cx="${cx - r * 0.48}" cy="${cy + r * 0.18}" rx="${r * 0.14}" ry="${r * 0.08}" fill="#f48a9c" opacity="0.6"/>`,
    `<ellipse cx="${cx + r * 0.48}" cy="${cy + r * 0.18}" rx="${r * 0.14}" ry="${r * 0.08}" fill="#f48a9c" opacity="0.6"/>`,
    `<path d="M ${cx - r * 0.1} ${cy + r * 0.15} q ${r * 0.1} ${r * 0.12} ${r * 0.2} 0" stroke="#3b3b3b" stroke-width="${r * 0.04}" fill="none" stroke-linecap="round"/>`,
  ].join("");
}

function character(who: Speaker, cx: number, cy: number, r: number): string {
  const { body, accent } = CHARACTER_STYLE[who];
  const stroke = `stroke="#4a4a4a" stroke-width="${r * 0.05}"`;
  let extra = "";
  if (who === "papi") {
    extra =
      `<ellipse cx="${cx - r * 0.45}" cy="${cy - r * 1.05}" rx="${r * 0.18}" ry="${r * 0.42}" fill="${body}" ${stroke}/>` +
      `<ellipse cx="${cx + r * 0.45}" cy="${cy - r * 1.05}" rx="${r * 0.18}" ry="${r * 0.42}" fill="${body}" ${stroke}/>`;
  } else if (who === "nezu") {
    extra =
      `<circle cx="${cx - r * 0.75}" cy="${cy - r * 0.8}" r="${r * 0.42}" fill="${body}" ${stroke}/>` +
      `<circle cx="${cx + r * 0.75}" cy="${cy - r * 0.8}" r="${r * 0.42}" fill="${body}" ${stroke}/>`;
  } else {
    extra =
      `<circle cx="${cx - r * 0.55}" cy="${cy - r * 0.75}" r="${r * 0.45}" fill="${body}" ${stroke}/>` +
      `<circle cx="${cx + r * 0.5}" cy="${cy - r * 0.8}" r="${r * 0.5}" fill="${body}" ${stroke}/>`;
  }
  return (
    extra +
    `<ellipse cx="${cx}" cy="${cy}" rx="${r}" ry="${r * 0.9}" fill="${body}" ${stroke}/>` +
    `<circle cx="${cx + r * 0.62}" cy="${cy + r * 0.5}" r="${r * 0.12}" fill="${accent}"/>` +
    face(cx, cy, r)
  );
}

/** Gradient background with soft shapes and a floor shadow. */
function backdrop(id: string, hue: number): string {
  return `<defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="hsl(${hue}, 85%, 93%)"/>
    <stop offset="1" stop-color="hsl(${(hue + 40) % 360}, 70%, 84%)"/>
  </linearGradient></defs>
  <rect width="${PANEL_W}" height="${PANEL_H}" fill="url(#${id})"/>
  <circle cx="860" cy="560" r="120" fill="#ffffff" opacity="0.35"/>
  <circle cx="150" cy="700" r="70" fill="#ffffff" opacity="0.35"/>
  <ellipse cx="512" cy="1265" rx="470" ry="70" fill="#000000" opacity="0.06"/>`;
}

const FRAME = `<rect x="6" y="6" width="${PANEL_W - 12}" height="${PANEL_H - 12}" rx="28" fill="none" stroke="#4a4a4a" stroke-width="12"/>`;

/** Older-job artwork for one panel (grey bars where the image model drew the words), at (offsetX, 0). */
function panelArtwork(n: number, spec: DemoPanelSpec, variant: number, offsetX: number): string {
  const hue = (n * 37 + variant * 140) % 360;
  const id = `g${n}_${variant}_${offsetX}`;
  const count = spec.characters.length;
  const r = count === 1 ? 210 : count === 2 ? 165 : 130;
  const xs = count === 1 ? [512] : count === 2 ? [330, 694] : [220, 512, 804];
  const bubbleLines = Math.min(3, Math.max(1, spec.dialogue.length + 1));
  const lines = Array.from({ length: bubbleLines }, (_, i) => {
    const w = i === bubbleLines - 1 ? 420 : 640;
    return `<rect x="${190}" y="${190 + i * 70}" width="${w}" height="26" rx="13" fill="#d8d3dc"/>`;
  }).join("");
  const bubbleBottom = 230 + bubbleLines * 70;
  const title = escapeXml(`${n} · ${spec.title}`);

  return `<g transform="translate(${offsetX} 0)">
  ${backdrop(id, hue)}
  <path d="M 300 ${bubbleBottom - 4} l 40 90 l 50 -92 z" fill="#ffffff" stroke="#4a4a4a" stroke-width="8" stroke-linejoin="round"/>
  <rect x="110" y="120" width="804" height="${bubbleBottom - 120}" rx="64" fill="#ffffff" stroke="#4a4a4a" stroke-width="8"/>
  <rect x="310" y="${bubbleBottom - 8}" width="72" height="18" fill="#ffffff"/>
  ${lines}
  ${spec.characters.map((who, i) => character(who, xs[i], 1100, r)).join("")}
  <rect x="60" y="1340" width="904" height="130" rx="30" fill="#ffffff" opacity="0.85"/>
  <text x="512" y="1424" text-anchor="middle" font-family="'Comic Neue','Comic Sans MS',sans-serif" font-size="52" font-weight="700" fill="#4a4a4a">${title}</text>
  ${FRAME}
</g>`;
}

// ---------------------------------------------------------------------------
// v3 artwork: the words typeset onto the picture, as the backend's lettering does
// ---------------------------------------------------------------------------

const INK = "#3b3b3b";
const LETTERING_FONT = "'Comic Sans MS','Chalkboard SE','Comic Neue',cursive";

/** Greedy word wrap by an estimated glyph width (good enough for placeholder art). */
function wrapWords(text: string, size: number, width: number): string[] {
  const max = Math.max(8, Math.floor(width / (size * 0.56)));
  const lines: string[] = [];
  for (const word of text.split(/\s+/).filter(Boolean)) {
    const last = lines[lines.length - 1];
    if (last !== undefined && last.length + 1 + word.length <= max) lines[lines.length - 1] = `${last} ${word}`;
    else lines.push(word);
  }
  return lines;
}

function lettering(
  lines: string[],
  x: number,
  y: number,
  size: number,
  { leading = size * 1.3, anchor = "start", weight = 400, fill = INK }: { leading?: number; anchor?: string; weight?: number; fill?: string } = {},
): string {
  return lines
    .map(
      (line, i) =>
        `<text x="${x}" y="${y + i * leading}" text-anchor="${anchor}" font-family="${LETTERING_FONT}" font-size="${size}" font-weight="${weight}" fill="${fill}">${escapeXml(line)}</text>`,
    )
    .join("");
}

/** Narration box across the top; returns the SVG and its bottom edge. */
function narrationBox(text: string, y: number): [string, number] {
  const size = 34;
  const leading = 46;
  const pad = 26;
  const lines = wrapWords(text, size, 912 - 2 * pad);
  const h = 2 * pad + (lines.length - 1) * leading + size;
  const box = `<rect x="56" y="${y}" width="912" height="${h}" fill="#fffbea" stroke="${INK}" stroke-width="5"/>`;
  return [box + lettering(lines, 56 + pad, y + pad + size * 0.8, size, { leading }), y + h];
}

/** Speech bubble with a short tail toward the speaker at `speakerX`; returns the SVG and its bottom edge. */
function speechBubble(text: string, x: number, y: number, w: number, speakerX: number): [string, number] {
  const size = 34;
  const leading = 44;
  const padY = 28;
  const lines = wrapWords(text, size, w - 88);
  const h = 2 * padY + (lines.length - 1) * leading + size;
  const bottom = y + h;
  const base = Math.min(Math.max(speakerX, x + 80), x + w - 80);
  const tip = base + Math.max(-40, Math.min(40, speakerX - base));
  const body = `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${Math.min(h / 2, 64)}" fill="#ffffff" stroke="${INK}" stroke-width="5"/>`;
  // An open path: its fill hides the bubble's outline where the tail joins, and only the tail's sides get a stroke.
  const tail = `<path d="M ${base - 22} ${bottom - 10} L ${tip} ${bottom + 66} L ${base + 22} ${bottom - 10}" fill="#ffffff" stroke="${INK}" stroke-width="5" stroke-linejoin="round"/>`;
  return [body + tail + lettering(lines, x + w / 2, y + padY + size * 0.8, size, { leading, anchor: "middle" }), bottom];
}

/** A bar chart in marker style: the paper's own bars in red, value at each bar's end, delta badge, source. */
function barChart(board: JobBoard, subtitle: string, x: number, y: number, w: number, bottom: number): string {
  const bars = (Array.isArray(board.bars) ? board.bars : []) as Array<{ label: string; value: string; highlight?: boolean }>;
  let svg = lettering([String(board.title ?? "")], x, y + 38, 38, { weight: 700 });
  let top = y + 60;
  if (subtitle) {
    const lines = wrapWords(subtitle, 26, w);
    svg += lettering(lines, x, top + 26, 26, { leading: 34, fill: "#6b6b6b" });
    top += 20 + lines.length * 34;
  }
  const values = bars.map((bar) => Number.parseFloat(bar.value));
  const max = Math.max(0, ...values);
  const labelW = 300;
  const barMax = w - labelW - 150;
  // Rows as tall as the board allows (up to a point), centred in the space under the title.
  const room = bottom - 70 - top;
  const rowH = Math.min(150, room / Math.max(bars.length, 1));
  top += (room - rowH * bars.length) / 2;
  bars.forEach((bar, i) => {
    const rowTop = top + 24 + i * rowH;
    const len = max > 0 ? (values[i] / max) * barMax : 0;
    const color = bar.highlight ? "#d94b4b" : "#5b8fd0";
    svg += lettering([bar.label], x + labelW - 18, rowTop + rowH / 2 + 10, 28, { anchor: "end" });
    svg += `<rect x="${x + labelW}" y="${rowTop + rowH * 0.2}" width="${len}" height="${rowH * 0.6}" fill="${color}" fill-opacity="0.45" stroke="${color}" stroke-width="4"/>`;
    svg += lettering([bar.value], x + labelW + len + 14, rowTop + rowH / 2 + 11, 30, { weight: 700, fill: color });
  });
  svg += `<line x1="${x + labelW}" y1="${top + 16}" x2="${x + labelW}" y2="${top + 32 + bars.length * rowH}" stroke="${INK}" stroke-width="4"/>`;
  // Delta badge: best highlighted bar vs best other bar, with the inputs' decimals (as the backend computes it).
  const best = (keep: boolean) => Math.max(...values.filter((v, i) => !!bars[i].highlight === keep && Number.isFinite(v)));
  const delta = best(true) - best(false);
  if (Number.isFinite(delta)) {
    const decimals = Math.max(...bars.map((bar) => (bar.value.split(".")[1] ?? "").length));
    const label = `${delta >= 0 ? "+" : "−"}${Math.abs(delta).toFixed(decimals)}`;
    svg += `<rect x="${x + w - 150}" y="${y + 4}" width="140" height="50" rx="25" fill="#ffffff" stroke="#d94b4b" stroke-width="4"/>`;
    svg += lettering([label], x + w - 80, y + 40, 30, { anchor: "middle", weight: 700, fill: "#d94b4b" });
  }
  // Grey footnotes, as the backend's board prints them: which way is better on the left, the source on the right.
  svg += lettering([board.higher_is_better === false ? "lower is better" : "higher is better"], x, bottom - 16, 24, { fill: "#8a8a8a" });
  if (board.source) svg += lettering([String(board.source)], x + w, bottom - 16, 24, { anchor: "end", fill: "#8a8a8a" });
  return svg;
}

/** v3 artwork: narration box at the top, bubbles below it, the cast underneath; board panels get a whiteboard chart. */
function typesetArtwork(n: number, spec: DemoPanelSpec, variant: number): string {
  const hue = (n * 37 + variant * 140) % 360;
  const board = BOARDS[n];
  const count = spec.characters.length;
  let words = "";
  let cast: string;
  if (board) {
    // Bubbles side by side above the board, the cast from the chest up below it; the narration is the chart's subtitle.
    const xs = count === 1 ? [512] : [300, 724];
    let top = 48;
    spec.dialogue.slice(0, 2).forEach(([speaker, line], i) => {
      const [svg, bottom] = speechBubble(line, i === 0 ? 40 : 544, 48, 440, xs[Math.max(0, spec.characters.indexOf(speaker))] ?? 512);
      words += svg;
      top = Math.max(top, bottom);
    });
    const boardTop = top + 100;
    words =
      `<rect x="64" y="${boardTop}" width="896" height="${1130 - boardTop}" rx="10" fill="#fdfdfb" stroke="#8b9099" stroke-width="12"/>` +
      barChart(board, spec.narration, 100, boardTop + 26, 824, 1130 - 18) +
      words;
    cast = spec.characters.map((who, i) => character(who, xs[i], 1330, 115)).join("");
  } else {
    const r = count === 1 ? 210 : count === 2 ? 165 : 130;
    const xs = count === 1 ? [512] : count === 2 ? [330, 694] : [220, 512, 804];
    let y = 48;
    if (spec.narration) {
      const [svg, bottom] = narrationBox(spec.narration, y);
      words += svg;
      y = bottom + 36;
    }
    for (const [speaker, line] of spec.dialogue) {
      const speakerX = xs[Math.max(0, spec.characters.indexOf(speaker))];
      const x = speakerX < 400 ? 48 : speakerX > 624 ? PANEL_W - 48 - 560 : (PANEL_W - 560) / 2;
      const [svg, bottom] = speechBubble(line, x, y, 560, speakerX);
      words += svg;
      y = bottom + 30;
    }
    cast = spec.characters.map((who, i) => character(who, xs[i], 1100, r)).join("");
  }
  return `${backdrop(`t${n}_${variant}`, hue)}${cast}${words}${FRAME}`;
}

function toDataUri(svg: string): string {
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

/** v3 panel picture (the single-layout demo is a v3 job). */
function singleImage(n: number, spec: DemoPanelSpec, variant: number): string {
  return toDataUri(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${PANEL_W}" height="${PANEL_H}" viewBox="0 0 ${PANEL_W} ${PANEL_H}">${typesetArtwork(n, spec, variant)}</svg>`,
  );
}

function stripImage(left: [number, DemoPanelSpec], right: [number, DemoPanelSpec], variant: number): string {
  const w = PANEL_W * 2;
  return toDataUri(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${PANEL_H}" viewBox="0 0 ${w} ${PANEL_H}">${panelArtwork(left[0], left[1], variant, 0)}${panelArtwork(right[0], right[1], variant, PANEL_W)}</svg>`,
  );
}

function combinedImage(title: string, panelCount: number): string {
  return toDataUri(
    `<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="2048" viewBox="0 0 1024 2048"><rect width="1024" height="2048" fill="#fff8e7"/><text x="512" y="980" text-anchor="middle" font-family="sans-serif" font-size="56" font-weight="700" fill="#4a4a4a">${escapeXml(title)}</text><text x="512" y="1060" text-anchor="middle" font-family="sans-serif" font-size="40" fill="#6b6b6b">Demo combined manga · ${panelCount} panels</text></svg>`,
  );
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const SPECS: DemoPanelSpec[] = CHAPTERS.flatMap((c) => c.panels);

/** v3 story beat of each panel, by panel number - 1. */
const BEATS = ["hook", "problem", "problem", "problem", "idea", "how", "how", "how", "how", "proof", "payoff", "recap"];

/** v3 board panels: the exhibit drawn on the board, by panel number (values from the paper's Table 2). */
const BOARDS: Record<number, JobBoard> = {
  10: {
    kind: "bar",
    title: "WMT 2014 English→German (BLEU)",
    higher_is_better: true,
    bars: [
      { label: "GNMT+RL ensemble", value: "26.30" },
      { label: "ConvS2S ensemble", value: "26.36" },
      { label: "Transformer (big)", value: "28.4", highlight: true },
    ],
    source: "Table 2",
  },
};

function chapterList(): JobChapter[] {
  let n = 0;
  return CHAPTERS.map((c) => ({
    id: c.id,
    title: c.title,
    panel_numbers: c.panels.map(() => ++n),
  }));
}

function chapterIdOf(n: number): string {
  let count = 0;
  for (const c of CHAPTERS) {
    count += c.panels.length;
    if (n <= count) return c.id;
  }
  return CHAPTERS[CHAPTERS.length - 1].id;
}

function basePanel(n: number, spec: DemoPanelSpec): JobPanel {
  return {
    n,
    chapter: chapterIdOf(n),
    title: spec.title,
    characters: [...spec.characters],
    dialogue: spec.dialogue.map(([speaker, text]) => ({ speaker, text })),
    narration: spec.narration,
    status: "done",
    image_url: null,
    width: null,
    height: null,
    attempts: 1,
    qa: null,
  };
}

const TITLE = "Attention Is All You Need";
const SUMMARY =
  "The Transformer replaces recurrence with attention: every word looks at every other word at once, which makes training faster and translation better.";

const PAPER_CARD: JobPaperCard = {
  problem:
    "The best translation models read a sentence one word at a time with recurrent networks. Every step waits for the previous one, so training can't be parallelized within a sentence, and links between distant words are hard to learn.",
  key_idea: "Drop recurrence entirely: let every word look directly at every other word through attention, all at once.",
  contributions: [
    "The Transformer: the first sequence-to-sequence model built entirely on attention, with no recurrence or convolutions.",
    "Scaled dot-product attention and multi-head attention as its core building blocks.",
    "Sinusoidal positional encodings that tell the model where each word sits.",
    "State-of-the-art results on two WMT 2014 translation benchmarks at a fraction of the previous training cost.",
  ],
  how: [
    "An encoder and a decoder, each a stack of 6 identical layers (model width 512, feed-forward width 2048), with residual connections and layer normalization.",
    "Scaled dot-product attention: softmax(QKᵀ / √dₖ) · V weighs every word's value by how well its key matches the query.",
    "Multi-head attention runs 8 heads of size 64 in parallel and combines their outputs.",
    "Sine and cosine positional encodings are added to the word embeddings to mark each word's position.",
    "Masked self-attention in the decoder lets each word look only at the words before it.",
  ],
  results: [
    "WMT 2014 English→German: 28.4 BLEU, more than 2 BLEU above the best previous results, ensembles included.",
    "WMT 2014 English→French: a new single-model state of the art after 3.5 days of training on 8 GPUs.",
    "The base model trains in 12 hours on 8 NVIDIA P100 GPUs and already reaches 27.3 BLEU on English→German.",
    "It also works beyond translation: 91.3 F1 on English constituency parsing, trained only on the WSJ part of the Penn Treebank.",
  ],
  limitations: [
    "Self-attention compares every pair of positions, so its cost grows with the square of the sequence length; restricting it to a local neighbourhood for very long inputs is left to future work.",
    "Averaging over attention-weighted positions lowers the effective resolution; multi-head attention is used to counter this.",
    "Output is still generated one word at a time; making generation less sequential is future work.",
  ],
  takeaway:
    "Attention alone is enough: the Transformer translates better than the best recurrent and convolutional models and trains significantly faster.",
};

/**
 * Key points of the demo paper. Core: P1–P6 and P8–P10; important: P7, P11–P15. The panels' `covers` miss
 * P10 (English→French), P7 (layer sizes) and P13 (parsing): those are only in the paper card.
 */
const COVERAGE: JobCoverage = { core: [8, 9], important: [4, 6] };

function createDemoJob(): JobState {
  const pending = new Set([11, 12]);
  const failed = 7;
  const panels = SPECS.map((spec, i): JobPanel => {
    const n = i + 1;
    const panel = basePanel(n, spec);
    const info = {
      detail: spec.detail,
      covers: [...spec.covers],
      kind: BOARDS[n] ? ("board" as const) : ("scene" as const),
      board: BOARDS[n] ?? null,
      beat: BEATS[n - 1],
    };
    if (pending.has(n)) return { ...panel, ...info, status: "pending", attempts: 0 };
    if (n === failed) {
      return {
        ...panel,
        ...info,
        status: "failed",
        attempts: 3,
        qa: { score: 3, text_exact: 0, text_total: 2, issues: ["Formula text is unreadable", "An extra character appeared"] },
      };
    }
    return {
      ...panel,
      ...info,
      image_url: singleImage(n, spec, 0),
      width: PANEL_W,
      height: PANEL_H,
      attempts: n === 4 ? 2 : 1,
      qa: { score: n === 4 ? 8 : 9, text_exact: spec.dialogue.length, text_total: spec.dialogue.length, issues: [] },
    };
  });
  const done = panels.filter((p) => p.status === "done").length;

  return {
    job_id: "demo_20261021_attention",
    status: "rendering",
    stage_message: "Drawing panel 11 of 12",
    created_at: "2026-10-21T12:05:01+08:00",
    updated_at: "2026-10-21T12:11:53+08:00",
    title: TITLE,
    summary: SUMMARY,
    language: "en-US",
    theme: "kumomo",
    layout: "single",
    progress: { done, total: panels.length, percent: Math.round((done / panels.length) * 100) },
    chapters: chapterList(),
    panels,
    combined_url: null,
    storyboard_url: null,
    cost: {
      images_generated: 12,
      image_tokens: 73728,
      text_prompt_tokens: 21480,
      text_completion_tokens: 6120,
      elapsed_seconds: 412.6,
    },
    error: null,
    paper_card: PAPER_CARD,
    coverage: COVERAGE,
    version: 3,
    quiz: { score: 0.917, questions: 12, failed_claim_ids: [4] },
  };
}

function createDemoStripJob(): JobState {
  const panels = SPECS.map((spec, i): JobPanel => {
    const n = i + 1;
    const index = (i % 2) as 0 | 1;
    const leftN = index === 0 ? n : n - 1;
    const url = stripImage([leftN, SPECS[leftN - 1]], [leftN + 1, SPECS[leftN]], 0);
    return {
      ...basePanel(n, spec),
      image_url: url,
      width: PANEL_W * 2,
      height: PANEL_H,
      qa: { score: 9, text_exact: spec.dialogue.length, text_total: spec.dialogue.length, issues: [] },
      strip_index: index,
      strip_width: PANEL_W * 2,
      strip_height: PANEL_H,
    };
  });

  return {
    job_id: "demo_20261021_attention_strip",
    status: "completed",
    stage_message: "Done",
    created_at: "2026-10-21T13:20:10+08:00",
    updated_at: "2026-10-21T13:27:44+08:00",
    title: `${TITLE} (strip layout)`,
    summary: SUMMARY,
    language: "en-US",
    theme: "kumomo",
    layout: "strip",
    progress: { done: panels.length, total: panels.length, percent: 100 },
    chapters: chapterList(),
    panels,
    combined_url: combinedImage(TITLE, panels.length),
    storyboard_url: null,
    cost: {
      images_generated: 6,
      image_tokens: 73728,
      text_prompt_tokens: 20950,
      text_completion_tokens: 5980,
      elapsed_seconds: 454.1,
    },
    error: null,
  };
}

function createDemoEarlyJob(): JobState {
  return {
    ...createDemoJob(),
    job_id: "demo_20261021_attention_early",
    status: "storyboarding",
    stage_message: "Writing the storyboard (chapter 2 of 3)",
    progress: { done: 0, total: 0, percent: 0 },
    chapters: [],
    panels: [],
    cost: { images_generated: 0, image_tokens: 0, text_prompt_tokens: 9800, text_completion_tokens: 1200, elapsed_seconds: 41.2 },
    paper_card: null,
    coverage: null,
    quiz: null,
  };
}

export const demoJob: JobState = createDemoJob();
export const demoStripJob: JobState = createDemoStripJob();
export const demoEarlyJob: JobState = createDemoEarlyJob();

/**
 * Simulated "regenerate": redraws panel `n` (and its strip partner, which
 * shares the image) with new placeholder art, as the backend would.
 */
export function redrawDemoPanel(job: JobState, n: number): JobState {
  const target = job.panels.find((p) => p.n === n);
  if (!target) return job;
  const strip = target.strip_index === 0 || target.strip_index === 1;
  const leftN = strip && target.strip_index === 1 ? n - 1 : n;
  const affected = strip ? new Set([leftN, leftN + 1]) : new Set([n]);
  const variant = target.attempts + 1;
  const stripUrl =
    strip && SPECS[leftN - 1] && SPECS[leftN]
      ? stripImage([leftN, SPECS[leftN - 1]], [leftN + 1, SPECS[leftN]], variant)
      : null;

  const panels = job.panels.map((p): JobPanel => {
    if (!affected.has(p.n)) return p;
    const spec = SPECS[p.n - 1];
    if (!spec) return p;
    return {
      ...p,
      status: "done",
      attempts: p.attempts + 1,
      image_url: stripUrl ?? singleImage(p.n, spec, variant),
      width: strip ? PANEL_W * 2 : PANEL_W,
      height: PANEL_H,
      qa: { score: 9, text_exact: spec.dialogue.length, text_total: spec.dialogue.length, issues: [] },
      error: null, // a successful render clears the previous error, as on the server
    };
  });
  const done = panels.filter((p) => p.status === "done").length;
  return {
    ...job,
    panels,
    progress: { done, total: panels.length, percent: Math.round((done / panels.length) * 100) },
  };
}
