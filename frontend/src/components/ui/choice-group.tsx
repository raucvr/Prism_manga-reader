"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

export interface ChoiceOption<T extends string> {
  value: T;
  label: string;
  hint?: string;
  icon?: React.ReactNode;
}

interface ChoiceGroupProps<T extends string> {
  legend: string;
  /** Optional helper text under the legend. */
  description?: string;
  name: string;
  value: T;
  options: ReadonlyArray<ChoiceOption<T>>;
  onChange: (value: T) => void;
  columns?: 2 | 3;
  className?: string;
}

/** A row of radio "cards" backed by native radio inputs (arrow keys work out of the box). */
export function ChoiceGroup<T extends string>({
  legend,
  description,
  name,
  value,
  options,
  onChange,
  columns = 3,
  className,
}: ChoiceGroupProps<T>) {
  const descriptionId = React.useId();
  return (
    <fieldset className={className} aria-describedby={description ? descriptionId : undefined}>
      <legend className="font-medium">{legend}</legend>
      {description && (
        <p id={descriptionId} className="mt-0.5 text-sm text-muted-foreground">
          {description}
        </p>
      )}
      <div className={cn("mt-2 grid gap-2", columns === 3 ? "grid-cols-3" : "grid-cols-2")}>
        {options.map((option) => {
          const checked = option.value === value;
          return (
            <label
              key={option.value}
              className={cn(
                "relative flex min-w-0 cursor-pointer flex-col items-center justify-center gap-1 rounded-2xl border-2 bg-white px-2 py-3 text-center transition-colors",
                "has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-primary-400 has-[:focus-visible]:ring-offset-2",
                checked ? "border-primary bg-primary/10" : "border-transparent hover:bg-primary/5",
              )}
            >
              <input
                type="radio"
                name={name}
                value={option.value}
                checked={checked}
                onChange={() => onChange(option.value)}
                className="sr-only"
              />
              {option.icon && <span aria-hidden="true">{option.icon}</span>}
              <span className="text-sm font-medium">{option.label}</span>
              {option.hint && <span className="text-xs text-muted-foreground">{option.hint}</span>}
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}
