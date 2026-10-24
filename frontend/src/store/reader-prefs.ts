import { create } from "zustand";
import { readJSON, writeJSON } from "@/lib/storage";

export type ReaderMode = "focus" | "scroll";
export type TextSize = "S" | "M" | "L";

export interface ReaderPrefs {
  mode: ReaderMode;
  textSize: TextSize;
  /** Fewer animations and a desaturated page background. */
  calm: boolean;
  /** Show dialogue/narration as text under each panel. */
  showText: boolean;
  /** Show each panel's "More detail" text (focus mode: the section is open). Off by default to keep the page calm. */
  showDetail: boolean;
  /** Reading language for jobs lettered in several languages (null: each job's own language). */
  language: string | null;
}

interface ReaderPrefsState extends ReaderPrefs {
  hydrated: boolean;
  /** Load saved prefs from localStorage (client only; call once after mount). */
  hydrate: () => void;
  setMode: (mode: ReaderMode) => void;
  setTextSize: (size: TextSize) => void;
  setCalm: (calm: boolean) => void;
  setShowText: (show: boolean) => void;
  setShowDetail: (show: boolean) => void;
  setLanguage: (language: string | null) => void;
}

const STORAGE_KEY = "prism.reader.prefs";

const DEFAULT_PREFS: ReaderPrefs = {
  mode: "focus",
  textSize: "M",
  calm: false,
  showText: true,
  showDetail: false,
  language: null,
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function sanitize(saved: Record<string, unknown> | null): ReaderPrefs {
  const prefs = { ...DEFAULT_PREFS };
  if (!saved) return prefs;
  if (saved.mode === "focus" || saved.mode === "scroll") prefs.mode = saved.mode;
  if (saved.textSize === "S" || saved.textSize === "M" || saved.textSize === "L") prefs.textSize = saved.textSize;
  if (typeof saved.calm === "boolean") prefs.calm = saved.calm;
  if (typeof saved.showText === "boolean") prefs.showText = saved.showText;
  if (typeof saved.showDetail === "boolean") prefs.showDetail = saved.showDetail;
  if (typeof saved.language === "string" && /^[a-z]{2}(-[A-Z]{2})?$/.test(saved.language)) prefs.language = saved.language;
  return prefs;
}

export const useReaderPrefs = create<ReaderPrefsState>((set, get) => {
  const update = (patch: Partial<ReaderPrefs>) => {
    set(patch);
    const { mode, textSize, calm, showText, showDetail, language, hydrated } = get();
    // Never overwrite saved prefs with defaults before they were loaded.
    if (hydrated) writeJSON(STORAGE_KEY, { mode, textSize, calm, showText, showDetail, language });
  };

  return {
    ...DEFAULT_PREFS,
    hydrated: false,
    hydrate: () => {
      if (get().hydrated) return;
      set({ ...sanitize(readJSON(STORAGE_KEY, isRecord)), hydrated: true });
    },
    setMode: (mode) => update({ mode }),
    setTextSize: (textSize) => update({ textSize }),
    setCalm: (calm) => update({ calm }),
    setShowText: (showText) => update({ showText }),
    setShowDetail: (showDetail) => update({ showDetail }),
    setLanguage: (language) => update({ language }),
  };
});

/** Tailwind classes for the reader's HTML text at each size. */
export const TEXT_SIZE_CLASS: Record<TextSize, string> = {
  S: "text-[15px] leading-relaxed",
  M: "text-lg leading-relaxed",
  L: "text-[22px] leading-relaxed",
};

/** One step smaller, for secondary reader text: the panels' "More detail" and the paper card. */
export const SECONDARY_TEXT_SIZE_CLASS: Record<TextSize, string> = {
  S: "text-sm leading-relaxed",
  M: "text-base leading-relaxed",
  L: "text-lg leading-relaxed",
};
