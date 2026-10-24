"use client";

import React, { useEffect, useState } from "react";
import { AlertTriangle, CircleSlash, Download, ImageOff, Loader2, RefreshCw, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { JobPanel } from "@/lib/api";
import { DEFAULT_PANEL_ASPECT, isStripPanel, panelErrorReason, redrawFailed, type ImageSize } from "@/lib/job-utils";
import { DialogueLine } from "@/components/reader/PanelText";

const ICON_BUTTON =
  "flex h-9 items-center justify-center rounded-full bg-white/90 text-foreground shadow ring-1 ring-black/5 transition-colors hover:bg-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400";

/** Small "redraw" button that asks for a second click, so a stray tap doesn't throw a panel away. */
export function RegenerateButton({ n, onRegenerate }: { n: number; onRegenerate: (n: number) => void }) {
  const [confirming, setConfirming] = useState(false);

  useEffect(() => {
    if (!confirming) return;
    const t = setTimeout(() => setConfirming(false), 5000);
    return () => clearTimeout(t);
  }, [confirming]);

  return (
    <button
      type="button"
      onClick={() => {
        if (confirming) {
          setConfirming(false);
          onRegenerate(n);
        } else {
          setConfirming(true);
        }
      }}
      aria-label={confirming ? `Confirm: redraw panel ${n}` : `Regenerate panel ${n}`}
      title={confirming ? "Click again to redraw" : "Not happy with this panel? Redraw it"}
      className={cn(ICON_BUTTON, confirming ? "gap-1.5 px-3 text-sm font-medium text-primary-700" : "w-9")}
    >
      <RefreshCw className="h-4 w-4" aria-hidden="true" />
      {confirming && <span>Redraw?</span>}
    </button>
  );
}

function PanelPlaceholder({
  panel,
  textSizeClass,
  speakerColors,
  lang,
  brokenImage,
  jobStopped,
  pendingWillDraw,
  onRegenerate,
  onRetryImage,
}: {
  panel: JobPanel;
  textSizeClass: string;
  speakerColors?: Map<string, string>;
  lang?: string;
  brokenImage: boolean;
  /** The job is no longer running (completed / failed / cancelled): redraws are allowed, pending panels will not be drawn on their own. */
  jobStopped: boolean;
  /** A pending panel will be drawn without user action. */
  pendingWillDraw: boolean;
  onRegenerate: (n: number) => void;
  onRetryImage: () => void;
}) {
  const failed = panel.status === "failed";
  const notDrawn = panel.status === "pending" && !pendingWillDraw;
  const problem = failed || brokenImage;
  const label = brokenImage
    ? "The picture didn't load"
    : failed
      ? "Couldn't draw this panel"
      : notDrawn
        ? "Not drawn: generation stopped"
        : panel.status === "qa"
          ? "Checking quality…"
          : panel.status === "pending"
            ? "Drawing soon…"
            : "Drawing…";
  const issue = failed && panel.qa?.available !== false ? panel.qa?.issues?.[0] : undefined;
  const reason = failed || notDrawn ? panelErrorReason(panel.error) : null;

  return (
    <div
      className={cn(
        "absolute inset-0 flex flex-col overflow-hidden p-4 sm:p-6",
        problem
          ? "bg-gradient-to-b from-red-50 to-orange-50"
          : "bg-gradient-to-b from-primary-50 via-white to-secondary-50",
      )}
    >
      {!problem && !notDrawn && <div className="skeleton-shimmer absolute inset-0" aria-hidden="true" />}
      <p
        className={cn(
          "relative flex items-center gap-2 text-sm font-semibold",
          problem ? "text-red-700" : "text-primary-700",
        )}
      >
        {brokenImage ? (
          <ImageOff className="h-4 w-4" aria-hidden="true" />
        ) : failed ? (
          <AlertTriangle className="h-4 w-4" aria-hidden="true" />
        ) : notDrawn ? (
          <CircleSlash className="h-4 w-4" aria-hidden="true" />
        ) : (
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
        )}
        {label}
      </p>
      <h3 lang={lang} className="relative mt-3 text-xl font-bold leading-snug sm:text-2xl">
        {panel.title}
      </h3>
      {panel.dialogue?.length > 0 && (
        <div lang={lang} className={cn("relative mt-3 min-h-0 flex-1 space-y-2 overflow-hidden", textSizeClass)}>
          {panel.dialogue.map((line, i) => (
            <DialogueLine key={i} line={line} colors={speakerColors} />
          ))}
        </div>
      )}
      {reason && <p className="relative mt-3 text-sm text-red-700">Last try failed: {reason}</p>}
      {issue && <p className="relative mt-3 text-sm text-red-700">Quality check: {issue}</p>}
      {failed && !jobStopped && (
        <p className="relative mt-3 text-center text-sm text-muted-foreground">
          {pendingWillDraw
            ? "You can redraw this panel once the others are finished."
            : "You can redraw this panel once the current redraw is done."}
        </p>
      )}
      {jobStopped && (failed || notDrawn) && (
        <Button onClick={() => onRegenerate(panel.n)} className="relative mt-4 shrink-0 gap-2 self-center">
          <RefreshCw className="h-4 w-4" aria-hidden="true" />
          {failed ? "Regenerate" : "Draw this panel"}
        </Button>
      )}
      {brokenImage && !failed && (
        <Button variant="outline" onClick={onRetryImage} className="relative mt-4 shrink-0 gap-2 self-center">
          <RefreshCw className="h-4 w-4" aria-hidden="true" />
          Try again
        </Button>
      )}
    </div>
  );
}

export interface PanelFrameProps {
  panel: JobPanel;
  /** Cache-busted image URL, or null when there is no picture yet. */
  src: string | null;
  /** Width / height of what is shown (a strip panel shows one half of its image). */
  aspect: number;
  /**
   * "contain": as large as possible inside the nearest size container (focus mode);
   * "width": full width of the column (scroll mode).
   */
  fit: "contain" | "width";
  textSizeClass: string;
  speakerColors?: Map<string, string>;
  /** Language of the story text (applied to the alt text and placeholder text). */
  lang?: string;
  /** The job is no longer running (completed / failed / cancelled): redraw buttons are shown. */
  jobStopped?: boolean;
  /** A pending panel will be drawn without user action (a full run, or a redraw that includes it). */
  pendingWillDraw?: boolean;
  downloadName: string;
  onRegenerate: (n: number) => void;
  onImageSize: (src: string, size: ImageSize) => void;
  lazy?: boolean;
  showNumber?: boolean;
  /** Id of the element that describes the picture (v3: its words as text, while that text is hidden). */
  describedBy?: string;
}

export function PanelFrame({
  panel,
  src,
  aspect,
  fit,
  textSizeClass,
  speakerColors,
  lang,
  jobStopped = false,
  pendingWillDraw = !jobStopped,
  downloadName,
  onRegenerate,
  onImageSize,
  lazy = false,
  showNumber = false,
  describedBy,
}: PanelFrameProps) {
  const [brokenSrc, setBrokenSrc] = useState<string | null>(null);
  const [reloads, setReloads] = useState(0);

  const ratio = Number.isFinite(aspect) && aspect > 0 ? aspect : DEFAULT_PANEL_ASPECT;
  const style: React.CSSProperties =
    fit === "contain"
      ? { aspectRatio: String(ratio), width: `min(100cqw, calc(100cqh * ${ratio.toFixed(5)}))` }
      : { aspectRatio: String(ratio), width: "100%" };

  const strip = isStripPanel(panel);
  const broken = !!src && brokenSrc === src;
  const showImage = !!src && !broken;
  const redrawing = panel.status === "rendering" || (panel.status === "pending" && pendingWillDraw);
  const [dismissedError, setDismissedError] = useState<string | null>(null);
  // A new redraw starts: its outcome is shown even if an earlier failure was dismissed.
  useEffect(() => {
    if (redrawing) setDismissedError(null);
  }, [redrawing]);
  const showRedrawFailure =
    showImage && !redrawing && panel.status !== "qa" && redrawFailed(panel) && panel.error !== dismissedError;
  const imgSrc =
    src && reloads > 0 && !src.startsWith("data:") ? `${src}${src.includes("?") ? "&" : "?"}reload=${reloads}` : src;
  const alt = panel.title || `Panel ${panel.n}`;

  return (
    <div
      className="relative overflow-hidden rounded-2xl bg-white shadow-lg ring-1 ring-black/5"
      style={style}
      aria-busy={redrawing || panel.status === "qa" || undefined}
    >
      {showImage && imgSrc ? (
        // eslint-disable-next-line @next/next/no-img-element -- images come from the backend / data URIs
        <img
          key={imgSrc}
          src={imgSrc}
          alt={alt}
          aria-describedby={describedBy}
          lang={lang}
          loading={lazy ? "lazy" : "eager"}
          decoding="async"
          draggable={false}
          onLoad={(e) => {
            const img = e.currentTarget;
            if (src && img.naturalWidth > 0 && img.naturalHeight > 0) {
              onImageSize(src, { w: img.naturalWidth, h: img.naturalHeight });
            }
          }}
          onError={() => setBrokenSrc(src)}
          className={cn(
            "absolute inset-0 h-full w-full transition-opacity",
            // Strip: the shared image is twice as wide as this frame; cover + position shows one half.
            strip ? "max-w-none object-cover" : "object-contain",
            redrawing && "opacity-50",
          )}
          style={strip ? { objectPosition: panel.strip_index === 1 ? "100% 50%" : "0% 50%" } : undefined}
        />
      ) : (
        <PanelPlaceholder
          panel={panel}
          textSizeClass={textSizeClass}
          speakerColors={speakerColors}
          lang={lang}
          brokenImage={broken}
          jobStopped={jobStopped}
          pendingWillDraw={pendingWillDraw}
          onRegenerate={onRegenerate}
          onRetryImage={() => {
            setBrokenSrc(null);
            setReloads((r) => r + 1);
          }}
        />
      )}

      {showImage && redrawing && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="flex items-center gap-2 rounded-full bg-white/95 px-4 py-2 text-sm font-medium shadow">
            <Loader2 className="h-4 w-4 animate-spin text-primary-500" aria-hidden="true" />
            Redrawing…
          </span>
        </div>
      )}

      {showImage && panel.status === "qa" && (
        <span className="absolute bottom-2 left-2 flex items-center gap-1.5 rounded-full bg-white/95 px-3 py-1 text-xs font-medium shadow">
          <Loader2 className="h-3.5 w-3.5 animate-spin text-secondary-600" aria-hidden="true" />
          Checking quality…
        </span>
      )}

      {showRedrawFailure && (
        <div
          role="status"
          className="absolute inset-x-2 bottom-2 flex items-start gap-2 rounded-2xl bg-white/95 p-2.5 text-sm shadow ring-1 ring-amber-200"
        >
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" aria-hidden="true" />
          <div className="min-w-0 flex-1">
            <p className="font-medium">Redraw failed. Kept the previous picture.</p>
            <p className="mt-0.5 text-muted-foreground">{panelErrorReason(panel.error)}</p>
          </div>
          {jobStopped && (
            <Button size="sm" variant="outline" onClick={() => onRegenerate(panel.n)} className="shrink-0 gap-1.5">
              <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
              Try again
            </Button>
          )}
          <button
            type="button"
            onClick={() => setDismissedError(panel.error ?? null)}
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full hover:bg-black/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400"
            aria-label="Hide this message"
            title="Hide"
          >
            <X className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>
      )}

      {showImage && panel.status === "failed" && !showRedrawFailure && (
        <div className="absolute inset-x-2 bottom-2 flex flex-wrap items-center justify-center gap-2 rounded-2xl bg-white/95 p-2 text-sm shadow">
          <AlertTriangle className="h-4 w-4 text-red-600" aria-hidden="true" />
          <span>This panel may have mistakes.</span>
          {jobStopped && (
            <Button size="sm" onClick={() => onRegenerate(panel.n)} className="gap-1.5">
              <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
              Regenerate
            </Button>
          )}
        </div>
      )}

      {showNumber && (
        <span
          className="absolute left-2 top-2 rounded-full bg-black/55 px-2.5 py-0.5 text-xs font-semibold text-white"
          aria-hidden="true"
        >
          {panel.n}
        </span>
      )}

      {showImage && panel.status === "done" && src && (
        <div className="absolute right-2 top-2 flex gap-1.5 opacity-80 transition-opacity focus-within:opacity-100 hover:opacity-100">
          <a
            href={src}
            download={downloadName}
            target="_blank"
            rel="noopener noreferrer"
            aria-label={`Download panel ${panel.n}`}
            title="Download this panel"
            className={cn(ICON_BUTTON, "w-9")}
          >
            <Download className="h-4 w-4" aria-hidden="true" />
          </a>
          {jobStopped && <RegenerateButton n={panel.n} onRegenerate={onRegenerate} />}
        </div>
      )}
    </div>
  );
}
