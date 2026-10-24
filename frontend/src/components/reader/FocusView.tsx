"use client";

import React, { useId, useLayoutEffect, useRef } from "react";
import { motion } from "framer-motion";
import { ChevronLeft, ChevronRight, RotateCcw } from "lucide-react";
import { cn } from "@/lib/utils";
import type { JobPanel } from "@/lib/api";
import { panelDetail, panelHasWords, type PaperCardSection, type PointCoverage } from "@/lib/job-utils";
import type { TextSize } from "@/store/reader-prefs";
import { PanelFrame } from "@/components/reader/PanelFrame";
import { PanelText } from "@/components/reader/PanelText";
import { DetailBox, DisclosureButton, PanelDetailToggle, useRevealOnOpen } from "@/components/reader/PanelDetail";
import { PaperCard, PaperCardHeading } from "@/components/reader/PaperCard";
import { panelTextElementId, type FocusPageRefs, type FrameHelpers } from "@/components/reader/types";

/**
 * Room (rem) kept for the text under the picture, so the picture doesn't jump in size between panels; never more
 * than the box's 38% cap. Panels with a detail add room for its "More detail" button, and for the detail when open.
 * v3 panels keep one row for their small toggles ("Text", "More detail"), plus room for each open section.
 */
const TEXT_RESERVE: Record<TextSize, number> = { S: 5, M: 6, L: 7.5 };
const DETAIL_BUTTON_RESERVE = 2.5;
const DETAIL_OPEN_RESERVE: Record<TextSize, number> = { S: 4.5, M: 5, L: 6 };
const TOGGLE_ROW_RESERVE = 2.5;
const TEXT_MAX_HEIGHT = "38%";

const SWIPE_MIN_PX = 50;
const SWIPE_MAX_MS = 800;

interface FocusViewProps {
  panels: JobPanel[];
  index: number;
  /** Show the paper card: the page after the last panel (`index` is then the last panel). */
  onCard: boolean;
  direction: 1 | -1;
  onPrev: () => void;
  onNext: () => void;
  onFirst: () => void;
  helpers: FrameHelpers;
  /** Show the panels' words as text under the picture (v3: the "Text" section is open). */
  showText: boolean;
  onShowTextChange: (show: boolean) => void;
  /**
   * v3 job: the words are typeset into each picture, so their text copy is a "Text" section, closed by default
   * (kept for screen readers, selection and larger type). Older jobs show the text whenever `showText` is on.
   */
  typeset: boolean;
  textSize: TextSize;
  /** The "More detail" section is open (a reader pref). */
  showDetail: boolean;
  onShowDetailChange: (show: boolean) => void;
  /** The paper card's sections (empty: no card page). */
  card: PaperCardSection[];
  coverage: PointCoverage | null;
  focusRefs: FocusPageRefs;
}

/** Removed, disabled or hidden: an element that had focus there has lost it (or is about to). */
function cannotHoldFocus(el: Element | null): boolean {
  if (!(el instanceof HTMLElement) || el === document.body) return false;
  return !el.isConnected || (el instanceof HTMLButtonElement && el.disabled) || el.getClientRects().length === 0;
}

export function FocusView({
  panels,
  index,
  onCard,
  direction,
  onPrev,
  onNext,
  onFirst,
  helpers,
  showText,
  onShowTextChange,
  typeset,
  textSize,
  showDetail,
  onShowDetailChange,
  card,
  coverage,
  focusRefs,
}: FocusViewProps) {
  const swipe = useRef<{ x: number; y: number; t: number; id: number } | null>(null);
  const cardTitleId = useId();
  const detailId = useId();
  const textRef = useRef<HTMLDivElement>(null);
  const textSection = useRevealOnOpen<HTMLDivElement>(showText);
  const detailSection = useRevealOnOpen<HTMLDivElement>(showDetail);
  const prevButton = useRef<HTMLButtonElement>(null);
  const nextButton = useRef<HTMLButtonElement>(null);
  const pageKey = onCard ? "card" : panels[index]?.n;
  const shownPage = useRef(pageKey);

  // A page change can remove or disable the control that had focus ("Next" on the last page, "Read the paper card",
  // the card's own buttons). Before paint, move focus to the new page instead of letting it drop to <body>: entering
  // the card focuses the card; otherwise the side button in the direction of travel takes it.
  useLayoutEffect(() => {
    const before = focusRefs.beforeNav.current;
    focusRefs.beforeNav.current = null;
    if (shownPage.current === pageKey) return;
    shownPage.current = pageKey;
    if (textRef.current) textRef.current.scrollTop = 0; // each panel's text starts at the top
    if (!before) return;
    const dropped = cannotHoldFocus(before.element);
    if (onCard) {
      const fromNowhere = !before.element || before.element === document.body;
      if (dropped || fromNowhere || before.inPage) focusRefs.card.current?.focus({ preventScroll: true });
    } else if (dropped && before.inPage) {
      const order = direction < 0 ? [prevButton.current, nextButton.current] : [nextButton.current, prevButton.current];
      order.find((button) => button && !cannotHoldFocus(button))?.focus({ preventScroll: true });
    }
  }, [pageKey, onCard, direction, focusRefs]);

  const panel = panels[index];
  if (!panel) return null;

  const src = helpers.src(panel);
  const hasCard = card.length > 0;
  const isFirst = index === 0 && !onCard;
  const isLastPanel = index === panels.length - 1;
  const detail = panelDetail(panel);
  // v3: words to put in the "Text" section (the dialogue only once the picture exists; until then the placeholder shows it).
  const words = typeset && panelHasWords(panel, !!src);
  const textId = panelTextElementId(panel.n);

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.pointerType === "mouse" || !e.isPrimary) return;
    swipe.current = { x: e.clientX, y: e.clientY, t: e.timeStamp, id: e.pointerId };
  };
  const onPointerUp = (e: React.PointerEvent) => {
    const start = swipe.current;
    swipe.current = null;
    if (!start || start.id !== e.pointerId) return;
    const dx = e.clientX - start.x;
    const dy = e.clientY - start.y;
    if (Math.abs(dx) < SWIPE_MIN_PX || Math.abs(dx) < Math.abs(dy) * 1.5) return;
    if (e.timeStamp - start.t > SWIPE_MAX_MS) return;
    if (dx < 0) onNext();
    else onPrev();
  };
  const onPointerCancel = () => {
    swipe.current = null;
  };

  const navButton =
    "absolute top-1/2 z-10 flex h-11 w-11 -translate-y-1/2 items-center justify-center rounded-full bg-white/85 text-foreground shadow-md ring-1 ring-black/5 transition hover:bg-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400 disabled:pointer-events-none disabled:opacity-0 sm:h-12 sm:w-12";

  const endNote = !isLastPanel ? null : hasCard ? (
    <EndNote message="That's the last panel.">
      <EndAction onClick={onNext}>
        Read the paper card
        <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
      </EndAction>
    </EndNote>
  ) : (
    <EndNote message="That's the last panel.">
      <EndAction onClick={onFirst}>
        <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
        Back to the start
      </EndAction>
    </EndNote>
  );

  // Text under the picture. Older jobs: dialogue + narration (when shown), then the panel's "More detail".
  // v3: a row of small toggles ("Text", "More detail"), each opening its section under the row.
  const textBlock = typeset ? words || !!detail : showText || !!detail;
  const textReserve = typeset
    ? (textBlock ? TOGGLE_ROW_RESERVE : 0) +
      (words && showText ? TEXT_RESERVE[textSize] : 0) +
      (detail && showDetail ? DETAIL_OPEN_RESERVE[textSize] : 0)
    : (showText ? TEXT_RESERVE[textSize] : 0) +
      (detail ? DETAIL_BUTTON_RESERVE + (showDetail ? DETAIL_OPEN_RESERVE[textSize] : 0) : 0);

  return (
    <div ref={focusRefs.page} className="flex min-h-0 flex-1 flex-col">
      {/* Picture (as large as the remaining space allows), or the paper card after the last panel */}
      <div className="relative min-h-0 flex-1">
        {onCard ? (
          <section
            ref={focusRefs.card}
            aria-labelledby={cardTitleId}
            tabIndex={0}
            className="absolute inset-0 overflow-y-auto overscroll-contain rounded-2xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400"
            style={{ touchAction: "pan-y pinch-zoom" }}
            onPointerDown={onPointerDown}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerCancel}
          >
            <motion.div
              initial={{ opacity: 0, x: direction * 24 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.18, ease: "easeOut" }}
              className="mx-auto max-w-2xl px-1 pb-4 pt-3 md:max-w-xl lg:max-w-2xl"
            >
              <PaperCardHeading id={cardTitleId} coverage={coverage} />
              <PaperCard sections={card} lang={helpers.lang} sizeClass={helpers.secondaryTextSizeClass} className="mt-4" />
              <EndNote message="You made it to the end!" className="mt-5">
                <EndAction onClick={onPrev}>
                  <ChevronLeft className="h-3.5 w-3.5" aria-hidden="true" />
                  Back to panel {panel.n}
                </EndAction>
                <EndAction onClick={onFirst}>
                  <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
                  Back to the start
                </EndAction>
              </EndNote>
            </motion.div>
          </section>
        ) : (
          <div
            className="absolute inset-0 flex select-none items-center justify-center [container-type:size]"
            style={{ touchAction: "pan-y pinch-zoom" }}
            onPointerDown={onPointerDown}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerCancel}
          >
            <motion.div
              key={panel.n}
              initial={{ opacity: 0, x: direction * 24 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.18, ease: "easeOut" }}
            >
              <PanelFrame
                panel={panel}
                src={src}
                aspect={helpers.aspect(panel)}
                fit="contain"
                textSizeClass={helpers.textSizeClass}
                speakerColors={helpers.speakerColors}
                lang={helpers.lang}
                jobStopped={helpers.jobStopped}
                pendingWillDraw={helpers.pendingWillDraw(panel)}
                downloadName={helpers.downloadName(panel)}
                onRegenerate={helpers.onRegenerate}
                onImageSize={helpers.onImageSize}
                // The closed "Text" section still describes the picture to screen readers.
                describedBy={src && words && !showText ? textId : undefined}
              />
            </motion.div>
          </div>
        )}

        {/* On the card page the side button would cover the text on narrow screens: it has its own "Back" button. */}
        <button
          ref={prevButton}
          type="button"
          onClick={onPrev}
          disabled={isFirst}
          aria-label={onCard ? `Back to panel ${panel.n}` : "Previous panel"}
          className={cn(navButton, "left-0", onCard && "hidden md:flex")}
        >
          <ChevronLeft className="h-6 w-6" aria-hidden="true" />
        </button>
        <button
          ref={nextButton}
          type="button"
          onClick={onNext}
          disabled={onCard || (isLastPanel && !hasCard)}
          aria-label={isLastPanel && hasCard && !onCard ? "Next: paper card" : "Next panel"}
          className={cn(navButton, "right-0")}
        >
          <ChevronRight className="h-6 w-6" aria-hidden="true" />
        </button>
      </div>

      {/* Dialogue + narration as selectable text, then "More detail" */}
      {!onCard && textBlock && (
        <div
          ref={textRef}
          className="shrink-0 overflow-y-auto pt-3"
          style={{ minHeight: `min(${textReserve}rem, ${TEXT_MAX_HEIGHT})`, maxHeight: TEXT_MAX_HEIGHT }}
        >
          <div className="mx-auto max-w-2xl px-1 pb-1">
            {typeset ? (
              <>
                <div className="-ml-1 flex flex-wrap items-center gap-x-2">
                  {words && (
                    <DisclosureButton
                      open={showText}
                      controls={textId}
                      onClick={() => {
                        if (!showText) textSection.reveal();
                        onShowTextChange(!showText);
                      }}
                    >
                      Text
                    </DisclosureButton>
                  )}
                  {detail && (
                    <DisclosureButton
                      open={showDetail}
                      controls={detailId}
                      onClick={() => {
                        if (!showDetail) detailSection.reveal();
                        onShowDetailChange(!showDetail);
                      }}
                    >
                      More detail
                    </DisclosureButton>
                  )}
                </div>
                {words && (
                  <div ref={textSection.ref} id={textId} hidden={!showText} className="mt-1.5">
                    <PanelText
                      panel={panel}
                      textSizeClass={helpers.textSizeClass}
                      speakerColors={helpers.speakerColors}
                      lang={helpers.lang}
                      includeDialogue={!!src}
                      narrationFirst
                    />
                  </div>
                )}
                {detail && (
                  <div ref={detailSection.ref} id={detailId} hidden={!showDetail} className="mt-2">
                    <DetailBox detail={detail} lang={helpers.lang} sizeClass={helpers.secondaryTextSizeClass} />
                  </div>
                )}
              </>
            ) : (
              <>
                {showText && (
                  <PanelText
                    panel={panel}
                    textSizeClass={helpers.textSizeClass}
                    speakerColors={helpers.speakerColors}
                    lang={helpers.lang}
                    includeDialogue={!!src}
                  />
                )}
                {detail && (
                  <PanelDetailToggle
                    detail={detail}
                    open={showDetail}
                    onOpenChange={onShowDetailChange}
                    lang={helpers.lang}
                    sizeClass={helpers.secondaryTextSizeClass}
                    className={showText ? "mt-2" : undefined}
                  />
                )}
              </>
            )}
            {endNote}
          </div>
        </div>
      )}
      {!onCard && !textBlock && endNote && <div className="shrink-0 pt-2">{endNote}</div>}
    </div>
  );
}

function EndNote({ message, className, children }: { message: string; className?: string; children: React.ReactNode }) {
  return (
    <p className={cn("mt-3 flex flex-wrap items-center justify-center gap-2 text-sm text-muted-foreground", className)}>
      <span>🎉 {message}</span>
      {children}
    </p>
  );
}

function EndAction({ onClick, children }: { onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex items-center gap-1 rounded-full px-2 py-1 font-medium text-primary-700 hover:bg-primary/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400"
    >
      {children}
    </button>
  );
}
