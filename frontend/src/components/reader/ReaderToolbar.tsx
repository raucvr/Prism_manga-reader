"use client";

import React, { useRef, useState } from "react";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { Captions, Check, Info, Leaf, NotebookText, RectangleVertical, Rows3, Type } from "lucide-react";
import { cn } from "@/lib/utils";
import type { JobPanel, JobState } from "@/lib/api";
import { languageLabel, type ChapterRun, type PaperCardSection, type PointCoverage } from "@/lib/job-utils";
import { SECONDARY_TEXT_SIZE_CLASS, useReaderPrefs, type TextSize } from "@/store/reader-prefs";
import { ChapterMenu, MENU_CONTENT, MENU_ITEM } from "@/components/reader/ChapterMenu";
import { DownloadMenu } from "@/components/reader/DownloadMenu";
import { PaperCardDialog } from "@/components/reader/PaperCard";

const ICON_BUTTON =
  "flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-foreground transition-colors hover:bg-primary/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400 data-[state=open]:bg-primary/10";
const SEGMENT_GROUP = "flex shrink-0 items-center rounded-full bg-white/80 p-0.5 ring-1 ring-black/5";
const SEGMENT =
  "flex h-8 min-w-8 items-center justify-center rounded-full px-2 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400";
const SEGMENT_ON = "bg-primary-200 text-primary-900";
const SEGMENT_OFF = "text-muted-foreground hover:bg-primary/10 hover:text-foreground";
const TOGGLE_ON = "bg-primary-200 text-primary-900 hover:bg-primary-200";
const LABELLED_BUTTON =
  "flex h-9 shrink-0 items-center justify-center gap-1.5 rounded-full px-2.5 text-sm font-medium text-foreground transition-colors hover:bg-primary/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400 aria-expanded:bg-primary/10";

const TEXT_SIZE_NAMES: Record<TextSize, string> = { S: "Small", M: "Medium", L: "Large" };
const TEXT_SIZES: TextSize[] = ["S", "M", "L"];

interface ReaderToolbarProps {
  job: JobState;
  /** Language of chapter / panel titles. */
  lang: string;
  panels: JobPanel[];
  /** Chapter sections (runs of consecutive panels in one chapter). */
  runs: ChapterRun[];
  index: number;
  /** Index into `runs` of the panel being read (-1 = none). */
  runIndex: number;
  onChapterSelect: (runIndex: number) => void;
  currentSrc: string | null;
  currentFileName: string;
  /** Focus mode is showing the paper card (the page after the last panel). */
  onCard: boolean;
  /** The paper card's sections (empty: no card, e.g. older jobs). */
  card: PaperCardSection[];
  coverage: PointCoverage | null;
  /** Some panel has a "More detail" text (otherwise its toggle is hidden). */
  anyDetail: boolean;
  /** Words as text under the panels (a reader pref for older jobs; v3 jobs keep it per visit, off at first). */
  showText: boolean;
  onShowTextChange: (show: boolean) => void;
  /** Languages the pictures are lettered in (more than one: a 中 / EN toggle); `lang` is the one shown. */
  languages?: string[];
  onLanguageChange?: (lang: string) => void;
}

export function ReaderToolbar({
  job,
  lang,
  panels,
  runs,
  index,
  runIndex,
  onChapterSelect,
  currentSrc,
  currentFileName,
  onCard,
  card,
  coverage,
  anyDetail,
  showText,
  onShowTextChange,
  languages = [],
  onLanguageChange,
}: ReaderToolbarProps) {
  const mode = useReaderPrefs((s) => s.mode);
  const textSize = useReaderPrefs((s) => s.textSize);
  const calm = useReaderPrefs((s) => s.calm);
  const showDetail = useReaderPrefs((s) => s.showDetail);
  const setMode = useReaderPrefs((s) => s.setMode);
  const setTextSize = useReaderPrefs((s) => s.setTextSize);
  const setCalm = useReaderPrefs((s) => s.setCalm);
  const setShowDetail = useReaderPrefs((s) => s.setShowDetail);

  const total = panels.length;
  const position = Math.min(index + 1, total);
  const panel = panels[index];

  // Paper card sheet: a toolbar button from `sm` up, an item of the display menu on narrow phones; neither while the
  // card page itself is showing. Focus goes back to whichever opened it.
  const [cardOpen, setCardOpen] = useState(false);
  const cardOpener = useRef<HTMLElement | null>(null);
  const displayMenuTrigger = useRef<HTMLButtonElement>(null);
  const cardFromMenu = useRef(false);
  const offerCard = card.length > 0 && !onCard;

  return (
    <div className="shrink-0">
      <div className="flex items-center gap-1 py-1.5 sm:gap-2">
        {/* Chapter (or title when there are no chapters) */}
        <div className="flex min-w-0 flex-1 items-center">
          {runs.length > 0 ? (
            <ChapterMenu runs={runs} panels={panels} currentRun={runIndex} onSelect={onChapterSelect} lang={lang} />
          ) : (
            <span lang={lang} className="truncate px-2 text-sm font-medium">
              {panel?.title}
            </span>
          )}
        </div>

        {/* 7 / 24 */}
        <span className="shrink-0 px-1 text-sm font-semibold tabular-nums text-muted-foreground" aria-hidden="true">
          {position} / {total}
        </span>
        <span className="sr-only" aria-live={mode === "focus" ? "polite" : "off"}>
          {onCard ? (
            "Paper card: a summary of the whole paper"
          ) : (
            <>
              Panel {position} of {total}
              {panel?.title ? `: ${panel.title}` : ""}
            </>
          )}
        </span>

        {/* Focus / scroll */}
        <div role="group" aria-label="Reading mode" className={SEGMENT_GROUP}>
          <button
            type="button"
            aria-pressed={mode === "focus"}
            aria-label="Focus mode: one panel at a time"
            title="Focus mode"
            onClick={() => setMode("focus")}
            className={cn(SEGMENT, mode === "focus" ? SEGMENT_ON : SEGMENT_OFF)}
          >
            <RectangleVertical className="h-4 w-4" aria-hidden="true" />
          </button>
          <button
            type="button"
            aria-pressed={mode === "scroll"}
            aria-label="Scroll mode: all panels in one column"
            title="Scroll mode"
            onClick={() => setMode("scroll")}
            className={cn(SEGMENT, mode === "scroll" ? SEGMENT_ON : SEGMENT_OFF)}
          >
            <Rows3 className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>

        {/* Reading language: the same pictures, lettered in each language */}
        {languages.length > 1 && onLanguageChange && (
          <div role="group" aria-label="Reading language" className={SEGMENT_GROUP}>
            {languages.map((l) => (
              <button
                key={l}
                type="button"
                lang={l}
                aria-pressed={lang === l}
                aria-label={l.startsWith("zh") ? "中文" : "English"}
                title={l.startsWith("zh") ? "中文" : "English"}
                onClick={() => onLanguageChange(l)}
                className={cn(SEGMENT, lang === l ? SEGMENT_ON : SEGMENT_OFF)}
              >
                {languageLabel(l)}
              </button>
            ))}
          </div>
        )}

        {/* Display settings, inline on wider screens */}
        <div className="hidden items-center gap-1 md:flex">
          <div role="group" aria-label="Text size" className={SEGMENT_GROUP}>
            {TEXT_SIZES.map((size) => (
              <button
                key={size}
                type="button"
                aria-pressed={textSize === size}
                aria-label={`${TEXT_SIZE_NAMES[size]} text`}
                title={`${TEXT_SIZE_NAMES[size]} text`}
                onClick={() => setTextSize(size)}
                className={cn(SEGMENT, textSize === size ? SEGMENT_ON : SEGMENT_OFF)}
              >
                {size}
              </button>
            ))}
          </div>
          <button
            type="button"
            aria-pressed={calm}
            aria-label="Calm mode: fewer animations, softer colours"
            title="Calm mode"
            onClick={() => setCalm(!calm)}
            className={cn(ICON_BUTTON, calm && TOGGLE_ON)}
          >
            <Leaf className="h-4 w-4" aria-hidden="true" />
          </button>
          <button
            type="button"
            aria-pressed={showText}
            aria-label="Show text under panels"
            title={showText ? "Hide text under panels" : "Show text under panels"}
            onClick={() => onShowTextChange(!showText)}
            className={cn(ICON_BUTTON, showText && TOGGLE_ON)}
          >
            <Captions className="h-4 w-4" aria-hidden="true" />
          </button>
          {anyDetail && (
            <button
              type="button"
              aria-pressed={showDetail}
              aria-label="Show more detail under panels"
              title={showDetail ? "Hide more detail under panels" : "Show more detail under panels"}
              onClick={() => setShowDetail(!showDetail)}
              className={cn(ICON_BUTTON, showDetail && TOGGLE_ON)}
            >
              <Info className="h-4 w-4" aria-hidden="true" />
            </button>
          )}
        </div>

        {/* Display settings menu on small screens */}
        <DropdownMenu.Root modal={false}>
          <DropdownMenu.Trigger asChild>
            <button
              ref={displayMenuTrigger}
              type="button"
              className={cn(ICON_BUTTON, "md:hidden")}
              aria-label={offerCard ? "Display settings and paper card" : "Display settings"}
              title="Display settings"
            >
              <Type className="h-4 w-4" aria-hidden="true" />
            </button>
          </DropdownMenu.Trigger>
          <DropdownMenu.Portal>
            <DropdownMenu.Content
              align="end"
              sideOffset={6}
              collisionPadding={12}
              className={MENU_CONTENT}
              style={{ width: "min(16rem, calc(100vw - 1.5rem))" }}
              onCloseAutoFocus={(e) => {
                // The paper card sheet is opening and takes the focus.
                if (cardFromMenu.current) e.preventDefault();
                cardFromMenu.current = false;
              }}
            >
              <DropdownMenu.Label className="px-3 py-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Text size
              </DropdownMenu.Label>
              <DropdownMenu.RadioGroup value={textSize} onValueChange={(v) => setTextSize(v as TextSize)}>
                {TEXT_SIZES.map((size) => (
                  <DropdownMenu.RadioItem
                    key={size}
                    value={size}
                    className={MENU_ITEM}
                    onSelect={(e) => e.preventDefault()}
                  >
                    <span className="flex-1">{TEXT_SIZE_NAMES[size]}</span>
                    <DropdownMenu.ItemIndicator>
                      <Check className="h-4 w-4 text-primary-600" aria-hidden="true" />
                    </DropdownMenu.ItemIndicator>
                  </DropdownMenu.RadioItem>
                ))}
              </DropdownMenu.RadioGroup>
              <DropdownMenu.Separator className="my-1 h-px bg-border" />
              <DropdownMenu.CheckboxItem
                checked={calm}
                onCheckedChange={(v) => setCalm(v === true)}
                onSelect={(e) => e.preventDefault()}
                className={MENU_ITEM}
              >
                <Leaf className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                <span className="flex-1">Calm mode</span>
                <DropdownMenu.ItemIndicator>
                  <Check className="h-4 w-4 text-primary-600" aria-hidden="true" />
                </DropdownMenu.ItemIndicator>
              </DropdownMenu.CheckboxItem>
              <DropdownMenu.CheckboxItem
                checked={showText}
                onCheckedChange={(v) => onShowTextChange(v === true)}
                onSelect={(e) => e.preventDefault()}
                className={MENU_ITEM}
              >
                <Captions className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                <span className="flex-1">Text under panels</span>
                <DropdownMenu.ItemIndicator>
                  <Check className="h-4 w-4 text-primary-600" aria-hidden="true" />
                </DropdownMenu.ItemIndicator>
              </DropdownMenu.CheckboxItem>
              {anyDetail && (
                <DropdownMenu.CheckboxItem
                  checked={showDetail}
                  onCheckedChange={(v) => setShowDetail(v === true)}
                  onSelect={(e) => e.preventDefault()}
                  className={MENU_ITEM}
                >
                  <Info className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                  <span className="flex-1">More detail under panels</span>
                  <DropdownMenu.ItemIndicator>
                    <Check className="h-4 w-4 text-primary-600" aria-hidden="true" />
                  </DropdownMenu.ItemIndicator>
                </DropdownMenu.CheckboxItem>
              )}
              {offerCard && (
                <>
                  <DropdownMenu.Separator className="my-1 h-px bg-border" />
                  <DropdownMenu.Item
                    onSelect={() => {
                      cardFromMenu.current = true;
                      cardOpener.current = displayMenuTrigger.current;
                      setCardOpen(true);
                    }}
                    className={MENU_ITEM}
                  >
                    <NotebookText className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                    <span className="flex-1">Paper card</span>
                  </DropdownMenu.Item>
                </>
              )}
            </DropdownMenu.Content>
          </DropdownMenu.Portal>
        </DropdownMenu.Root>

        {offerCard && (
          <button
            type="button"
            aria-haspopup="dialog"
            aria-expanded={cardOpen}
            title="Paper card: the whole paper at a glance"
            onClick={(e) => {
              cardOpener.current = e.currentTarget;
              setCardOpen(true);
            }}
            className={cn(LABELLED_BUTTON, "hidden sm:flex")}
          >
            <NotebookText className="h-4 w-4 shrink-0" aria-hidden="true" />
            Paper card
          </button>
        )}
        {card.length > 0 && (
          <PaperCardDialog
            open={cardOpen}
            onOpenChange={setCardOpen}
            returnFocusRef={cardOpener}
            sections={card}
            coverage={coverage}
            lang={lang}
            sizeClass={SECONDARY_TEXT_SIZE_CLASS[textSize]}
          />
        )}

        <DownloadMenu
          job={job}
          panel={panel}
          panelSrc={currentSrc}
          panelFileName={currentFileName}
          triggerClassName={ICON_BUTTON}
        />
      </div>

      {/* Reading progress */}
      <div
        role="progressbar"
        aria-label="Reading progress"
        aria-valuemin={1}
        aria-valuemax={Math.max(total, 1)}
        aria-valuenow={position}
        aria-valuetext={`Panel ${position} of ${total}`}
        className="h-1 overflow-hidden rounded-full bg-gray-200/80"
      >
        <div
          className="h-full rounded-full bg-primary-400 transition-[width] duration-300"
          style={{ width: `${total > 0 ? (position / total) * 100 : 0}%` }}
        />
      </div>
    </div>
  );
}
