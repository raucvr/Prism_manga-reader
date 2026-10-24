"use client";

import React, { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import type { JobPanel, JobState } from "@/lib/api";
import {
  buildChapterRuns,
  buildImageVersions,
  buildSpeakerColors,
  defaultAspect,
  isTypesetJob,
  jobLanguages,
  localizeJob,
  localizePanel,
  panelAspect,
  panelDetail,
  panelFileName,
  paperCardSections,
  pointCoverage,
  versionedUrl,
  type ImageSize,
  isTerminal,
  readingLanguage,
} from "@/lib/job-utils";
import { readReaderPosition, writeReaderPosition } from "@/lib/storage";
import { SECONDARY_TEXT_SIZE_CLASS, TEXT_SIZE_CLASS, useReaderPrefs } from "@/store/reader-prefs";
import { ReaderToolbar } from "@/components/reader/ReaderToolbar";
import { FocusView } from "@/components/reader/FocusView";
import { ScrollView } from "@/components/reader/ScrollView";
import {
  chapterElementId,
  panelElementId,
  type FocusPageRefs,
  type FrameHelpers,
  type NavFocus,
} from "@/components/reader/types";

export interface ReaderHandle {
  /** Show panel `n` (focus mode) or scroll to it (scroll mode). */
  goToPanel: (n: number) => void;
}

interface ReaderProps {
  job: JobState;
  /** Status / error line shown above the toolbar. */
  banner?: React.ReactNode;
  onRegenerate: (n: number) => void;
}

function prefersReducedMotion(): boolean {
  try {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
}

/** Keys typed into these elements must not turn pages. */
const TYPING_TARGETS = "input, textarea, select, [contenteditable=''], [contenteditable='true'], [role='menu'], [role='dialog']";

/** ↑ / ↓ step (px) when the reader scrolls the paper card page for the keyboard. */
const CARD_LINE_PX = 48;

export const Reader = forwardRef<ReaderHandle, ReaderProps>(function Reader({ job: sourceJob, banner, onRegenerate }, ref) {
  // Bilingual v3 jobs letter the same pictures in both languages: the reader shows the chosen one (saved as a pref).
  const preferredLanguage = useReaderPrefs((s) => s.language);
  const setPreferredLanguage = useReaderPrefs((s) => s.setLanguage);
  const languages = useMemo(() => jobLanguages(sourceJob), [sourceJob]);
  const lang = readingLanguage(sourceJob, preferredLanguage);
  const job = useMemo(() => localizeJob(sourceJob, lang), [sourceJob, lang]);
  const mode = useReaderPrefs((s) => s.mode);
  const textSize = useReaderPrefs((s) => s.textSize);
  const showTextPref = useReaderPrefs((s) => s.showText);
  const setShowTextPref = useReaderPrefs((s) => s.setShowText);
  const showDetail = useReaderPrefs((s) => s.showDetail);
  const setShowDetail = useReaderPrefs((s) => s.setShowDetail);
  const calm = useReaderPrefs((s) => s.calm);
  const textSizeClass = TEXT_SIZE_CLASS[textSize];
  const secondaryTextSizeClass = SECONDARY_TEXT_SIZE_CLASS[textSize];

  // v3 pictures carry their words, typeset in code, so the text copy under them starts closed and is opened for this
  // visit only; older jobs (words drawn by the image model) follow the saved "text under panels" pref.
  const typeset = isTypesetJob(job);
  const [typesetTextOpen, setTypesetTextOpen] = useState(false);
  const showText = typeset ? typesetTextOpen : showTextPref;
  const setShowText = typeset ? setTypesetTextOpen : setShowTextPref;

  const panels = useMemo(
    () => [...sourceJob.panels].map((p) => localizePanel(p, lang, sourceJob.language)).sort((a, b) => a.n - b.n),
    [sourceJob.panels, sourceJob.language, lang],
  );
  const chapters = useMemo(() => job.chapters ?? [], [job.chapters]);
  const { runs, runOf } = useMemo(() => buildChapterRuns(chapters, panels), [chapters, panels]);
  const versions = useMemo(() => buildImageVersions(panels), [panels]);
  const fallbackAspect = useMemo(() => defaultAspect(panels), [panels]);
  const speakerColors = useMemo(() => buildSpeakerColors(panels), [panels]);
  const [naturalSizes, setNaturalSizes] = useState<Record<string, ImageSize>>({});

  // Information layer, shown as text (absent in older jobs): per-panel details and a card summarising the paper.
  const anyDetail = useMemo(() => panels.some((p) => panelDetail(p) !== ""), [panels]);
  const card = useMemo(() => paperCardSections(job.paper_card), [job.paper_card]);
  const coverage = useMemo(() => pointCoverage(job.coverage?.core), [job.coverage]);

  // Focus mode puts keyboard focus back on the page when a page change removes the focused control.
  const pageRef = useRef<HTMLDivElement>(null);
  const cardScrollRef = useRef<HTMLElement>(null);
  const navFocus = useRef<NavFocus | null>(null);
  const focusRefs: FocusPageRefs = useMemo(() => ({ page: pageRef, card: cardScrollRef, beforeNav: navFocus }), []);

  const [index, setIndex] = useState(0);
  const [direction, setDirection] = useState<1 | -1>(1);
  const [restored, setRestored] = useState(false);

  const total = panels.length;
  // Focus mode pages: the panels, then the paper card (page `total`) when the job has one.
  const lastPage = Math.max(card.length > 0 ? total : total - 1, 0);
  const page = Math.min(Math.max(index, 0), lastPage);
  const onCard = card.length > 0 && page === total;
  const safeIndex = Math.min(page, Math.max(total - 1, 0));
  const current: JobPanel | undefined = panels[safeIndex];
  const currentN = current?.n;
  const runIndex = currentN !== undefined ? runOf.get(currentN) ?? -1 : -1;

  // ---- Remember the position per job ---------------------------------------
  useEffect(() => {
    if (restored || total === 0) return;
    const saved = readReaderPosition(job.job_id);
    const savedIndex = saved !== null ? panels.findIndex((p) => p.n === saved) : -1;
    if (savedIndex > 0) setIndex(savedIndex);
    setRestored(true);
  }, [restored, total, panels, job.job_id]);

  useEffect(() => {
    if (restored && currentN !== undefined) writeReaderPosition(job.job_id, currentN);
  }, [restored, currentN, job.job_id]);

  // ---- Navigation ----------------------------------------------------------
  const goTo = useCallback(
    (target: number) => {
      if (total === 0) return;
      const next = Math.min(Math.max(target, 0), lastPage);
      const active = document.activeElement;
      navFocus.current = { element: active, inPage: !!active && !!pageRef.current?.contains(active) };
      setDirection(next >= page ? 1 : -1);
      setIndex(next);
    },
    [total, lastPage, page],
  );
  const next = useCallback(() => goTo(page + 1), [goTo, page]);
  const prev = useCallback(() => goTo(page - 1), [goTo, page]);
  const first = useCallback(() => goTo(0), [goTo]);

  const scrollToElement = useCallback(
    (id: string) => {
      document.getElementById(id)?.scrollIntoView({
        behavior: calm || prefersReducedMotion() ? "auto" : "smooth",
        block: "start",
      });
    },
    [calm],
  );

  const goToPanel = useCallback(
    (n: number) => {
      const i = panels.findIndex((p) => p.n === n);
      if (i < 0) return;
      goTo(i);
      if (mode === "scroll") scrollToElement(panelElementId(n));
    },
    [panels, goTo, mode, scrollToElement],
  );

  useImperativeHandle(ref, () => ({ goToPanel }), [goToPanel]);

  /** Jump to the first panel of a chapter section (a run of consecutive panels in one chapter). */
  const selectRun = useCallback(
    (ri: number) => {
      const run = runs[ri];
      if (!run) return;
      goTo(run.start);
      if (mode === "scroll") scrollToElement(chapterElementId(ri));
    },
    [runs, goTo, mode, scrollToElement],
  );

  // Focus mode is laid out to fit the viewport: start it at the top of the page.
  useEffect(() => {
    if (mode === "focus") window.scrollTo({ top: 0 });
  }, [mode]);

  // Keyboard: ← / → (and PageUp/PageDown), Space / Shift+Space in focus mode.
  // On the paper card page the page keys and ↑ / ↓ scroll the card instead; ← goes back to the last panel.
  useEffect(() => {
    if (mode !== "focus") return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
      const target = e.target instanceof HTMLElement ? e.target : null;
      if (target?.closest(TYPING_TARGETS)) return;
      const space = e.key === " " || e.key === "Spacebar";
      const box = cardScrollRef.current;
      if (onCard && box && (space || ["PageDown", "PageUp", "ArrowDown", "ArrowUp"].includes(e.key))) {
        if (space && target?.closest("button, a, [role='button']")) return;
        // Focus inside the card: the browser scrolls it.
        if (target && box.contains(target)) return;
        const pageStep = box.clientHeight * 0.85;
        const back = e.key === "PageUp" || e.key === "ArrowUp" || (space && e.shiftKey);
        const step = e.key === "ArrowDown" || e.key === "ArrowUp" ? CARD_LINE_PX : pageStep;
        e.preventDefault();
        box.scrollBy({ top: back ? -step : step, behavior: calm || prefersReducedMotion() ? "auto" : "smooth" });
        return;
      }
      if (e.key === "ArrowRight" || e.key === "PageDown") {
        e.preventDefault();
        next();
      } else if (e.key === "ArrowLeft" || e.key === "PageUp") {
        e.preventDefault();
        prev();
      } else if (space) {
        // On a focused button Space already "clicks" it; don't turn the page twice.
        if (target?.closest("button, a, [role='button']")) return;
        e.preventDefault();
        if (e.shiftKey) prev();
        else next();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [mode, next, prev, onCard, calm]);

  // ---- Images --------------------------------------------------------------
  const srcOf = useCallback(
    (p: JobPanel) => (p.image_url ? versionedUrl(p.image_url, versions.get(p.image_url)) : null),
    [versions],
  );

  const onImageSize = useCallback((src: string, size: ImageSize) => {
    setNaturalSizes((prevSizes) => {
      const known = prevSizes[src];
      return known && known.w === size.w && known.h === size.h ? prevSizes : { ...prevSizes, [src]: size };
    });
  }, []);

  const helpers: FrameHelpers = useMemo(() => {
    const running = !isTerminal(job.status);
    // While a user-triggered redraw runs, only the panels being redrawn get drawn.
    const redrawing = new Set(job.redraw_panels ?? []);
    return {
      src: srcOf,
      aspect: (p: JobPanel) => {
        const src = srcOf(p);
        return panelAspect(p, src ? naturalSizes[src] : undefined, fallbackAspect);
      },
      downloadName: (p: JobPanel) => panelFileName(job, p),
      onRegenerate,
      onImageSize,
      textSizeClass,
      secondaryTextSizeClass,
      speakerColors,
      lang: job.language,
      jobStopped: !running,
      pendingWillDraw: (p: JobPanel) => running && (redrawing.size === 0 || redrawing.has(p.n)),
    };
  }, [
    srcOf,
    naturalSizes,
    fallbackAspect,
    job,
    onRegenerate,
    onImageSize,
    textSizeClass,
    secondaryTextSizeClass,
    speakerColors,
  ]);

  // Warm the cache for the next panels so turning the page feels instant.
  useEffect(() => {
    if (mode !== "focus") return;
    for (const i of [safeIndex + 1, safeIndex + 2, safeIndex - 1]) {
      const p = panels[i];
      const src = p ? srcOf(p) : null;
      if (!src || src.startsWith("data:")) continue;
      const img = new window.Image();
      img.decoding = "async";
      img.src = src;
    }
  }, [mode, safeIndex, panels, srcOf]);

  if (total === 0 || !current) return null;

  const toolbar = (
    <ReaderToolbar
      job={job}
      lang={job.language}
      panels={panels}
      runs={runs}
      index={safeIndex}
      runIndex={runIndex}
      onChapterSelect={selectRun}
      currentSrc={srcOf(current)}
      currentFileName={panelFileName(job, current)}
      onCard={mode === "focus" && onCard}
      card={card}
      coverage={coverage}
      anyDetail={anyDetail}
      showText={showText}
      onShowTextChange={setShowText}
      languages={languages}
      onLanguageChange={setPreferredLanguage}
    />
  );

  if (mode === "focus") {
    return (
      <section aria-label="Manga reader" className="reader-focus mx-auto flex w-full max-w-5xl flex-col px-3 pb-2 sm:px-4">
        {banner && <div className="shrink-0 space-y-2 pb-1">{banner}</div>}
        {toolbar}
        {restored && (
          <FocusView
            panels={panels}
            index={safeIndex}
            onCard={onCard}
            direction={direction}
            onPrev={prev}
            onNext={next}
            onFirst={first}
            helpers={helpers}
            showText={showText}
            onShowTextChange={setShowText}
            typeset={typeset}
            textSize={textSize}
            showDetail={showDetail}
            onShowDetailChange={setShowDetail}
            card={card}
            coverage={coverage}
            focusRefs={focusRefs}
          />
        )}
      </section>
    );
  }

  return (
    <section aria-label="Manga reader" className="mx-auto w-full max-w-5xl px-3 pb-24 sm:px-4">
      {banner && <div className="space-y-2 pb-1">{banner}</div>}
      <div className="sticky top-0 z-20 -mx-3 bg-white/80 px-3 pb-1 backdrop-blur-md sm:-mx-4 sm:px-4">{toolbar}</div>
      {restored && (
        <ScrollView
          panels={panels}
          runs={runs}
          runOf={runOf}
          initialIndex={page}
          onCurrentChange={setIndex}
          helpers={helpers}
          showText={showText}
          typeset={typeset}
          showDetail={showDetail}
          card={card}
          coverage={coverage}
        />
      )}
    </section>
  );
});
