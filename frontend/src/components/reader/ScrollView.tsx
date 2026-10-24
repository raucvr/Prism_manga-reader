"use client";

import React, { Fragment, useEffect, useId, useRef } from "react";
import type { JobPanel } from "@/lib/api";
import { panelDetail, panelHasWords, type ChapterRun, type PaperCardSection, type PointCoverage } from "@/lib/job-utils";
import { PanelFrame } from "@/components/reader/PanelFrame";
import { PanelText } from "@/components/reader/PanelText";
import { PanelDetailText } from "@/components/reader/PanelDetail";
import { PaperCard, PaperCardHeading } from "@/components/reader/PaperCard";
import {
  PAPER_CARD_ELEMENT_ID,
  chapterElementId,
  panelElementId,
  panelTextElementId,
  type FrameHelpers,
} from "@/components/reader/types";

interface ScrollViewProps {
  panels: JobPanel[];
  /** Chapter sections: each run of consecutive panels in one chapter gets a heading. */
  runs: ChapterRun[];
  /** Panel number -> index into `runs`. */
  runOf: Map<number, number>;
  /** Panel to bring into view when this view opens (`panels.length`: the paper card). */
  initialIndex: number;
  /** Reports the panel in the middle of the screen while scrolling. */
  onCurrentChange: (index: number) => void;
  helpers: FrameHelpers;
  /** Show each panel's words as text under it. */
  showText: boolean;
  /** v3 job: the words are typeset into each picture (their text copy, when not shown, still describes it). */
  typeset: boolean;
  /** Show each panel's "More detail" text under it. */
  showDetail: boolean;
  /** The paper card's sections, shown after the last panel (empty: no card). */
  card: PaperCardSection[];
  coverage: PointCoverage | null;
}

export function ScrollView({
  panels,
  runs,
  runOf,
  initialIndex,
  onCurrentChange,
  helpers,
  showText,
  typeset,
  showDetail,
  card,
  coverage,
}: ScrollViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cardTitleId = useId();
  const panelsRef = useRef(panels);
  const onChangeRef = useRef(onCurrentChange);
  const panelKey = panels.map((p) => p.n).join(",");

  // Latest values for the observer callback (which outlives renders).
  useEffect(() => {
    panelsRef.current = panels;
    onChangeRef.current = onCurrentChange;
  });

  // Start where the reader was (e.g. when switching from focus mode).
  const initialIndexRef = useRef(initialIndex);
  useEffect(() => {
    const panel = panelsRef.current[initialIndexRef.current];
    if (panel && initialIndexRef.current > 0) {
      document.getElementById(panelElementId(panel.n))?.scrollIntoView({ block: "start" });
    } else if (!panel && initialIndexRef.current > 0) {
      // Coming from the paper card page of focus mode.
      document.getElementById(PAPER_CARD_ELEMENT_ID)?.scrollIntoView({ block: "start" });
    }
  }, []);

  // Track which panel is in the middle band of the viewport.
  useEffect(() => {
    const container = containerRef.current;
    if (!container || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          const n = Number((entry.target as HTMLElement).dataset.panel);
          const index = panelsRef.current.findIndex((p) => p.n === n);
          if (index >= 0) onChangeRef.current(index);
        }
      },
      { rootMargin: "-40% 0px -55% 0px" },
    );
    container.querySelectorAll<HTMLElement>("[data-panel]").forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [panelKey]);

  return (
    <div ref={containerRef} className="mx-auto w-full max-w-[720px] space-y-6 pb-8 pt-2">
      {panels.map((panel, i) => {
        const runIndex = runOf.get(panel.n) ?? -1;
        const run = runs[runIndex];
        const heading = run && run.start === i ? run : null;
        const src = helpers.src(panel);
        const detail = showDetail ? panelDetail(panel) : "";
        // v3 with the text hidden: the copy stays in the page, hidden, as the picture's description for screen readers.
        const describe = typeset && !showText && !!src && panelHasWords(panel, true);

        return (
          <Fragment key={panel.n}>
            {heading && (
              <h2 id={chapterElementId(runIndex)} className="scroll-mt-28 pt-6 text-center">
                <span className="block text-sm font-semibold uppercase tracking-wide text-primary-700">
                  Chapter {runIndex + 1}
                </span>
                <span lang={helpers.lang} className="mt-1 block text-2xl font-bold">
                  {heading.title}
                </span>
              </h2>
            )}
            <article
              id={panelElementId(panel.n)}
              data-panel={panel.n}
              aria-label={`Panel ${panel.n}: ${panel.title}`}
              className="scroll-mt-28"
            >
              <PanelFrame
                panel={panel}
                src={src}
                aspect={helpers.aspect(panel)}
                fit="width"
                lazy
                showNumber
                textSizeClass={helpers.textSizeClass}
                speakerColors={helpers.speakerColors}
                lang={helpers.lang}
                jobStopped={helpers.jobStopped}
                pendingWillDraw={helpers.pendingWillDraw(panel)}
                downloadName={helpers.downloadName(panel)}
                onRegenerate={helpers.onRegenerate}
                onImageSize={helpers.onImageSize}
                describedBy={describe ? panelTextElementId(panel.n) : undefined}
              />
              {showText && (
                <PanelText
                  panel={panel}
                  textSizeClass={helpers.textSizeClass}
                  speakerColors={helpers.speakerColors}
                  lang={helpers.lang}
                  includeDialogue={!!src}
                  narrationFirst={typeset}
                  className="mt-3 px-1"
                />
              )}
              {describe && (
                <div id={panelTextElementId(panel.n)} hidden>
                  <PanelText
                    panel={panel}
                    textSizeClass={helpers.textSizeClass}
                    lang={helpers.lang}
                    narrationFirst
                  />
                </div>
              )}
              {detail && (
                <PanelDetailText
                  detail={detail}
                  lang={helpers.lang}
                  sizeClass={helpers.secondaryTextSizeClass}
                  className="mx-1 mt-3"
                />
              )}
            </article>
          </Fragment>
        );
      })}

      {card.length > 0 && (
        <section id={PAPER_CARD_ELEMENT_ID} aria-labelledby={cardTitleId} className="scroll-mt-28 pt-6">
          <PaperCardHeading id={cardTitleId} coverage={coverage} />
          <PaperCard sections={card} lang={helpers.lang} sizeClass={helpers.secondaryTextSizeClass} className="mt-4" />
        </section>
      )}
    </div>
  );
}
