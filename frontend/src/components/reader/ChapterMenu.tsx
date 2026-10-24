"use client";

import React, { useMemo } from "react";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { Check, ChevronDown, List } from "lucide-react";
import type { JobPanel } from "@/lib/api";
import type { ChapterRun } from "@/lib/job-utils";

export const MENU_CONTENT =
  "z-50 overflow-y-auto rounded-2xl bg-white p-1.5 text-foreground shadow-xl ring-1 ring-black/5 focus:outline-none";
export const MENU_ITEM =
  "flex cursor-pointer select-none items-start gap-3 rounded-xl px-3 py-2 text-sm outline-none data-[disabled]:cursor-default data-[highlighted]:bg-primary/10 data-[disabled]:opacity-50";

interface ChapterMenuProps {
  /** Chapter sections in reading order (a chapter split into several runs is listed once per run). */
  runs: ChapterRun[];
  panels: JobPanel[];
  /** Index into `runs` of the panel being read (-1 = none). */
  currentRun: number;
  /** Jump to the first panel of a run. */
  onSelect: (runIndex: number) => void;
  /** Language of the chapter titles. */
  lang?: string;
}

/** Shows the current chapter; opens a list to jump to any chapter. */
export function ChapterMenu({ runs, panels, currentRun, onSelect, lang }: ChapterMenuProps) {
  const statusByPanel = useMemo(() => new Map(panels.map((p) => [p.n, p.status] as const)), [panels]);
  const current = runs[currentRun];

  if (runs.length === 0) return null;

  return (
    <DropdownMenu.Root modal={false}>
      <DropdownMenu.Trigger asChild>
        <button
          type="button"
          className="flex h-9 min-w-0 items-center gap-1.5 rounded-full px-2.5 text-sm font-medium hover:bg-primary/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400 data-[state=open]:bg-primary/10"
          aria-label={current ? `Chapters. Current chapter: ${current.title}` : "Chapters"}
        >
          <List className="h-4 w-4 shrink-0" aria-hidden="true" />
          <span lang={current ? lang : undefined} className="min-w-0 truncate">
            {current ? current.title : "Chapters"}
          </span>
          <ChevronDown className="h-4 w-4 shrink-0 opacity-60" aria-hidden="true" />
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          align="start"
          sideOffset={6}
          collisionPadding={12}
          className={MENU_CONTENT}
          style={{ width: "min(22rem, calc(100vw - 1.5rem))", maxHeight: "min(70vh, 28rem)" }}
        >
          <DropdownMenu.Label className="px-3 py-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Jump to a chapter
          </DropdownMenu.Label>
          <DropdownMenu.RadioGroup value={String(currentRun)} onValueChange={(value) => onSelect(Number(value))}>
            {runs.map((run, i) => {
              const count = run.panelNumbers.length;
              const ready = run.panelNumbers.filter((n) => statusByPanel.get(n) === "done").length;
              return (
                <DropdownMenu.RadioItem key={i} value={String(i)} className={MENU_ITEM}>
                  <span
                    className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary-100 text-xs font-bold text-primary-800"
                    aria-hidden="true"
                  >
                    {i + 1}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span lang={lang} className="block font-medium">
                      {run.title}
                    </span>
                    <span className="block text-xs text-muted-foreground">
                      {count} {count === 1 ? "panel" : "panels"}
                      {ready < count ? ` · ${ready} ready` : ""}
                    </span>
                  </span>
                  <DropdownMenu.ItemIndicator>
                    <Check className="mt-1 h-4 w-4 text-primary-600" aria-hidden="true" />
                  </DropdownMenu.ItemIndicator>
                </DropdownMenu.RadioItem>
              );
            })}
          </DropdownMenu.RadioGroup>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
