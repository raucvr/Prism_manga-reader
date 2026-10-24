"use client";

import React from "react";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { BookOpen, Download, Image as ImageIcon } from "lucide-react";
import type { JobPanel, JobState } from "@/lib/api";
import { combinedFileName } from "@/lib/job-utils";
import { MENU_CONTENT, MENU_ITEM } from "@/components/reader/ChapterMenu";

interface DownloadMenuProps {
  job: JobState;
  /** The panel being read and its (cache-busted) image URL. */
  panel: JobPanel | undefined;
  panelSrc: string | null;
  panelFileName: string;
  triggerClassName: string;
}

export function DownloadMenu({ job, panel, panelSrc, panelFileName, triggerClassName }: DownloadMenuProps) {
  const combined = job.combined_url;

  return (
    <DropdownMenu.Root modal={false}>
      <DropdownMenu.Trigger asChild>
        <button type="button" className={triggerClassName} aria-label="Download" title="Download">
          <Download className="h-4 w-4" aria-hidden="true" />
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          align="end"
          sideOffset={6}
          collisionPadding={12}
          className={MENU_CONTENT}
          style={{ width: "min(19rem, calc(100vw - 1.5rem))" }}
        >
          {combined ? (
            <DropdownMenu.Item asChild className={MENU_ITEM}>
              <a href={combined} download={combinedFileName(job, combined)} target="_blank" rel="noopener noreferrer">
                <BookOpen className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                <span>
                  <span className="block font-medium">Whole manga</span>
                  <span className="block text-xs text-muted-foreground">All panels in one long image</span>
                </span>
              </a>
            </DropdownMenu.Item>
          ) : (
            <DropdownMenu.Item disabled className={MENU_ITEM}>
              <BookOpen className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
              <span>
                <span className="block font-medium">Whole manga</span>
                <span className="block text-xs text-muted-foreground">Available when every panel is finished</span>
              </span>
            </DropdownMenu.Item>
          )}

          {panel && panelSrc ? (
            <DropdownMenu.Item asChild className={MENU_ITEM}>
              <a href={panelSrc} download={panelFileName} target="_blank" rel="noopener noreferrer">
                <ImageIcon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                <span>
                  <span className="block font-medium">This panel</span>
                  <span className="block text-xs text-muted-foreground">
                    Panel {panel.n}
                    {panel.strip_index === 0 || panel.strip_index === 1 ? " (with its strip partner)" : ""}
                  </span>
                </span>
              </a>
            </DropdownMenu.Item>
          ) : (
            <DropdownMenu.Item disabled className={MENU_ITEM}>
              <ImageIcon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
              <span>
                <span className="block font-medium">This panel</span>
                <span className="block text-xs text-muted-foreground">Not drawn yet</span>
              </span>
            </DropdownMenu.Item>
          )}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
