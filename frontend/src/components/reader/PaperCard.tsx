"use client";

import React from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle,
  BarChart3,
  Cog,
  Flag,
  HelpCircle,
  Lightbulb,
  ListChecks,
  NotebookText,
  Sparkles,
  X,
  type LucideIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { textLang, type PaperCardSection, type PaperCardSectionId, type PointCoverage } from "@/lib/job-utils";
import { useReaderPrefs } from "@/store/reader-prefs";

const SECTION_ICONS: Record<PaperCardSectionId, LucideIcon> = {
  problem: HelpCircle,
  key_idea: Lightbulb,
  contributions: Sparkles,
  how: Cog,
  results: BarChart3,
  limitations: AlertTriangle,
  takeaway: Flag,
};

/** The two things worth remembering get a soft tint (plain white in calm mode). */
const SECTION_TINT: Partial<Record<PaperCardSectionId, string>> = {
  key_idea: "bg-primary-50 ring-primary-200",
  takeaway: "bg-secondary-50 ring-secondary-200",
};

/** "Covers 8 of 9 core points" */
export function CoverageLine({ coverage, className }: { coverage: PointCoverage; className?: string }) {
  return (
    <p
      className={cn("flex items-center gap-1.5 text-sm text-muted-foreground", className)}
      title="Core points are what you need to understand the paper. Covered ones are explained in the panels."
    >
      <ListChecks className="h-4 w-4 shrink-0" aria-hidden="true" />
      Covers {coverage.covered} of {coverage.total} core points
    </p>
  );
}

/** Title and coverage line of the card shown as a page (focus mode) or as the last section (scroll mode). */
export function PaperCardHeading({
  id,
  coverage,
  className,
}: {
  id: string;
  coverage: PointCoverage | null;
  className?: string;
}) {
  const calm = useReaderPrefs((s) => s.calm);
  return (
    <div className={cn("flex flex-col items-center text-center", className)}>
      <h2 id={id} className="flex items-center gap-2 text-2xl font-bold">
        <NotebookText className={cn("h-6 w-6", calm ? "text-muted-foreground" : "text-primary-500")} aria-hidden="true" />
        Paper card
      </h2>
      {coverage && <CoverageLine coverage={coverage} className="mt-1" />}
    </div>
  );
}

interface PaperCardProps {
  sections: PaperCardSection[];
  /** The job's language (an untranslated English fallback is marked as English). */
  lang?: string;
  /** Body text size classes (follows the reader's text size). */
  sizeClass: string;
  className?: string;
}

/** The paper card's sections, each under its own h3 (the card's title is an h2). */
export function PaperCard({ sections, lang, sizeClass, className }: PaperCardProps) {
  const calm = useReaderPrefs((s) => s.calm);
  return (
    <div className={cn("space-y-3", className)}>
      {sections.map((section) => {
        const Icon = SECTION_ICONS[section.id];
        return (
          <div
            key={section.id}
            className={cn("rounded-2xl bg-white/85 p-4 shadow-sm ring-1 ring-black/5", !calm && SECTION_TINT[section.id])}
          >
            <h3 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-primary-700">
              <Icon className={cn("h-4 w-4 shrink-0", calm ? "text-muted-foreground" : "text-primary-500")} aria-hidden="true" />
              {section.label}
            </h3>
            <div className={cn("mt-2 select-text break-words", sizeClass)}>
              {section.list ? (
                <ul className="list-disc space-y-1.5 pl-5 marker:text-primary-300">
                  {section.items.map((item, i) => (
                    <li key={i} lang={textLang(item, lang)}>
                      {item}
                    </li>
                  ))}
                </ul>
              ) : (
                section.items.map((item, i) => (
                  <p key={i} lang={textLang(item, lang)} className={i > 0 ? "mt-2" : undefined}>
                    {item}
                  </p>
                ))
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

interface PaperCardDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Gets focus back when the sheet closes: the toolbar button or menu that opened it. */
  returnFocusRef: React.RefObject<HTMLElement | null>;
  sections: PaperCardSection[];
  coverage: PointCoverage | null;
  lang?: string;
  sizeClass: string;
}

/** The paper card in a side sheet, opened from the toolbar on any panel without losing the reader's place. */
export function PaperCardDialog({
  open,
  onOpenChange,
  returnFocusRef,
  sections,
  coverage,
  lang,
  sizeClass,
}: PaperCardDialogProps) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <AnimatePresence>
        {open && (
          <Dialog.Portal forceMount>
            <Dialog.Overlay asChild forceMount>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 z-40 bg-black/30"
              />
            </Dialog.Overlay>
            <Dialog.Content
              asChild
              forceMount
              aria-describedby={undefined}
              onCloseAutoFocus={(e) => {
                e.preventDefault();
                returnFocusRef.current?.focus();
              }}
            >
              <motion.div
                initial={{ x: "100%" }}
                animate={{ x: 0 }}
                exit={{ x: "100%" }}
                transition={{ type: "spring", damping: 25, stiffness: 200 }}
                // The sheet itself scrolls, so ↑ / ↓ / Page keys work while the close button has focus.
                className="fixed right-0 top-0 z-50 h-full w-full max-w-lg overflow-y-auto overscroll-contain bg-cream shadow-2xl focus:outline-none"
              >
                <div className="sticky top-0 z-10 flex items-start justify-between gap-3 bg-cream px-5 pb-3 pt-5 sm:px-6">
                  <div className="min-w-0">
                    <Dialog.Title className="text-2xl font-bold">Paper card</Dialog.Title>
                    {coverage && <CoverageLine coverage={coverage} className="mt-1" />}
                  </div>
                  <Dialog.Close asChild>
                    <Button variant="ghost" size="icon" className="shrink-0" aria-label="Close the paper card">
                      <X className="h-5 w-5" aria-hidden="true" />
                    </Button>
                  </Dialog.Close>
                </div>
                <PaperCard sections={sections} lang={lang} sizeClass={sizeClass} className="px-5 pb-8 pt-1 sm:px-6" />
              </motion.div>
            </Dialog.Content>
          </Dialog.Portal>
        )}
      </AnimatePresence>
    </Dialog.Root>
  );
}
