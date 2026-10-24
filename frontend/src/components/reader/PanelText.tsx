"use client";

import React from "react";
import { cn } from "@/lib/utils";
import type { JobDialogueLine, JobPanel } from "@/lib/api";
import { panelBoard, speakerColor, speakerName, type BoardSummary } from "@/lib/job-utils";

/** "Speaker: text", with a stable colour per speaker. */
export function DialogueLine({ line, colors }: { line: JobDialogueLine; colors?: Map<string, string> }) {
  const name = speakerName(line.speaker);
  return (
    <p>
      {name && <span className={cn("font-bold", speakerColor(line.speaker, colors))}>{name}: </span>}
      {line.text}
    </p>
  );
}

/**
 * A board panel's exhibit as text, laid out like the board: title, one line per bar / row / step / item (highlighted
 * ones in bold), then the grey footnotes (unit or note on the left, source on the right).
 */
function BoardText({ board }: { board: BoardSummary }) {
  return (
    <div className="rounded-2xl bg-white/70 px-3.5 py-2.5 ring-1 ring-black/5">
      {board.title && <p className="font-semibold">{board.title}</p>}
      {board.subtitle && <p className="text-muted-foreground">{board.subtitle}</p>}
      {board.lines.length > 0 && (
        <ul className={cn("space-y-0.5", (board.title || board.subtitle) && "mt-1")}>
          {board.lines.map((line, i) => (
            <li key={i} className={cn(line.strong && "font-semibold", line.muted && "text-sm text-muted-foreground")}>
              {line.text}
            </li>
          ))}
        </ul>
      )}
      {(board.caption || board.source) && (
        <p className="mt-1 flex flex-wrap justify-between gap-x-3 text-sm text-muted-foreground">
          <span>{board.caption}</span>
          <span>{board.source}</span>
        </p>
      )}
    </div>
  );
}

interface PanelTextProps {
  panel: JobPanel;
  textSizeClass: string;
  /** False while the "drawing…" placeholder already shows the dialogue. */
  includeDialogue?: boolean;
  /** List the narration first, as a v3 picture reads (its narration box is at the top). */
  narrationFirst?: boolean;
  speakerColors?: Map<string, string>;
  lang?: string;
  className?: string;
}

/** The panel's dialogue, narration and board exhibit as real, selectable HTML text. */
export function PanelText({
  panel,
  textSizeClass,
  includeDialogue = true,
  narrationFirst = false,
  speakerColors,
  lang,
  className,
}: PanelTextProps) {
  const dialogue = includeDialogue ? panel.dialogue ?? [] : [];
  const narration = panel.narration?.trim();
  const board = panelBoard(panel, lang);
  if (dialogue.length === 0 && !narration && !board) return null;

  const narrationLine = narration && (
    <p className="border-l-4 border-secondary-300 pl-3 italic text-muted-foreground">{narration}</p>
  );
  return (
    <div lang={lang} className={cn("select-text space-y-2 break-words", textSizeClass, className)}>
      {narrationFirst && narrationLine}
      {dialogue.map((line, i) => (
        <DialogueLine key={i} line={line} colors={speakerColors} />
      ))}
      {!narrationFirst && narrationLine}
      {board && <BoardText board={board} />}
    </div>
  );
}
