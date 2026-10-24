import type { MutableRefObject, RefObject } from "react";
import type { JobPanel } from "@/lib/api";
import type { ImageSize } from "@/lib/job-utils";

/** Per-panel data and callbacks shared by the focus and scroll views. */
export interface FrameHelpers {
  /** Cache-busted image URL, or null while there is no picture. */
  src: (panel: JobPanel) => string | null;
  /** Width / height to display (strip panels: one half of the image). */
  aspect: (panel: JobPanel) => number;
  downloadName: (panel: JobPanel) => string;
  onRegenerate: (n: number) => void;
  onImageSize: (src: string, size: ImageSize) => void;
  textSizeClass: string;
  /** One step smaller than `textSizeClass`: the panels' "More detail" and the paper card. */
  secondaryTextSizeClass: string;
  speakerColors: Map<string, string>;
  /** Language of the story text (e.g. "zh-CN"), for correct CJK glyphs and screen-reader voices. */
  lang: string;
  /** The job is no longer running (completed / failed / cancelled): redraws are allowed and pending panels won't be drawn on their own. */
  jobStopped: boolean;
  /** This panel, while pending, will be drawn without user action (false during a redraw of other panels). */
  pendingWillDraw: (panel: JobPanel) => boolean;
}

/** What had focus when the reader changed page. */
export interface NavFocus {
  element: Element | null;
  /** It was on the focus-mode page (picture or card, text, side buttons), not in the toolbar or a menu. */
  inPage: boolean;
}

/** Shared by the reader and the focus view so keyboard focus stays on the page when a page change removes it. */
export interface FocusPageRefs {
  /** The focus-mode page. */
  page: RefObject<HTMLDivElement>;
  /** Scroll box of the paper card page (the reader scrolls it with the page keys). */
  card: RefObject<HTMLElement>;
  /** Set by the reader on every page change, read and cleared by the focus view. */
  beforeNav: MutableRefObject<NavFocus | null>;
}

export const panelElementId = (n: number) => `prism-panel-${n}`;
/** The panel's words as text (v3: also the picture's description for screen readers while that text is hidden). */
export const panelTextElementId = (n: number) => `prism-panel-text-${n}`;
export const chapterElementId = (index: number) => `prism-chapter-${index}`;
export const PAPER_CARD_ELEMENT_ID = "prism-paper-card";
