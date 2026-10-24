"use client";

import React, { useEffect, useId, useRef } from "react";
import { ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { textLang } from "@/lib/job-utils";

/** Soft box around a panel's detail: the precise facts behind the panel, shown as text and never drawn. */
const DETAIL_BOX = "select-text break-words rounded-2xl bg-white/70 px-3.5 py-2.5 text-muted-foreground ring-1 ring-black/5";

interface DisclosureButtonProps {
  open: boolean;
  /** Id of the section the button opens. */
  controls: string;
  onClick: () => void;
  className?: string;
  children: React.ReactNode;
}

/** Small pill button that opens or closes a section under the picture ("Text", "More detail"). */
export function DisclosureButton({ open, controls, onClick, className, children }: DisclosureButtonProps) {
  return (
    <button
      type="button"
      aria-expanded={open}
      aria-controls={controls}
      onClick={onClick}
      className={cn(
        "inline-flex items-center gap-1 rounded-full py-1 pl-1 pr-2.5 text-sm font-medium text-primary-700 hover:bg-primary/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400",
        className,
      )}
    >
      <ChevronRight className={cn("h-4 w-4 transition-transform", open && "rotate-90")} aria-hidden="true" />
      {children}
    </button>
  );
}

/**
 * For a section opened with its button: the box under the picture may be scrolled, so bring the section into view.
 * Call `reveal()` in the click handler that opens it; opening it any other way (a toolbar toggle) doesn't scroll.
 */
export function useRevealOnOpen<T extends HTMLElement>(open: boolean) {
  const ref = useRef<T>(null);
  const pending = useRef(false);
  useEffect(() => {
    if (open && pending.current) ref.current?.scrollIntoView({ block: "nearest" });
    pending.current = false;
  }, [open]);
  return {
    ref,
    reveal: () => {
      pending.current = true;
    },
  };
}

interface PanelDetailProps {
  /** The panel's detail (non-empty). */
  detail: string;
  /** The job's language (an untranslated English fallback is marked as English). */
  lang?: string;
  /** Text size classes, one step smaller than the dialogue. */
  sizeClass: string;
  className?: string;
}

/** The detail in its soft box. */
export function DetailBox({ detail, lang, sizeClass, className }: PanelDetailProps) {
  return (
    <p lang={textLang(detail, lang)} className={cn(DETAIL_BOX, sizeClass, className)}>
      {detail}
    </p>
  );
}

interface PanelDetailToggleProps extends PanelDetailProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/** Focus mode: a "More detail" section under the panel's text. Open/closed is a reader pref, so it carries over. */
export function PanelDetailToggle({ detail, lang, sizeClass, className, open, onOpenChange }: PanelDetailToggleProps) {
  const contentId = useId();
  const { ref, reveal } = useRevealOnOpen<HTMLDivElement>(open);

  return (
    <div className={className}>
      <DisclosureButton
        open={open}
        controls={contentId}
        onClick={() => {
          if (!open) reveal();
          onOpenChange(!open);
        }}
        className="-ml-1"
      >
        More detail
      </DisclosureButton>
      <div ref={ref} id={contentId} hidden={!open} className="mt-1.5">
        <DetailBox detail={detail} lang={lang} sizeClass={sizeClass} />
      </div>
    </div>
  );
}

/** Scroll mode: the detail as small text under the panel. */
export function PanelDetailText({ detail, lang, sizeClass, className }: PanelDetailProps) {
  return (
    <div className={cn(DETAIL_BOX, sizeClass, className)}>
      <p className="select-none text-xs font-semibold uppercase tracking-wide">More detail</p>
      <p lang={textLang(detail, lang)} className="mt-0.5">
        {detail}
      </p>
    </div>
  );
}
