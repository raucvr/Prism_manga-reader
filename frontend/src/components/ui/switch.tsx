"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

interface SwitchProps {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  label: string;
  description?: string;
  disabled?: boolean;
  className?: string;
}

/** Labelled on/off switch built on a native checkbox (keyboard + screen-reader friendly). */
export function Switch({ checked, onCheckedChange, label, description, disabled, className }: SwitchProps) {
  const id = React.useId();
  const descriptionId = description ? `${id}-description` : undefined;

  return (
    <label
      htmlFor={id}
      className={cn(
        "flex cursor-pointer items-center justify-between gap-4 rounded-2xl p-3 transition-colors hover:bg-primary/5",
        disabled && "cursor-not-allowed opacity-60",
        className,
      )}
    >
      <span className="min-w-0">
        <span className="block font-medium">{label}</span>
        {description && (
          <span id={descriptionId} className="mt-0.5 block text-sm text-muted-foreground">
            {description}
          </span>
        )}
      </span>
      <input
        id={id}
        type="checkbox"
        role="switch"
        className="peer sr-only"
        checked={checked}
        disabled={disabled}
        aria-describedby={descriptionId}
        onChange={(e) => onCheckedChange(e.target.checked)}
      />
      <span
        aria-hidden="true"
        className={cn(
          "relative inline-flex h-7 w-12 shrink-0 rounded-full transition-colors",
          "peer-focus-visible:ring-2 peer-focus-visible:ring-primary-400 peer-focus-visible:ring-offset-2",
          checked ? "bg-primary-400" : "bg-gray-300",
        )}
      >
        <span
          className={cn(
            "absolute left-0.5 top-0.5 h-6 w-6 rounded-full bg-white shadow transition-transform",
            checked && "translate-x-5",
          )}
        />
      </span>
    </label>
  );
}
