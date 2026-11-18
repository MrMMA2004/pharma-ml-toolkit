import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** The shadcn/ui class-merge helper (the one utility every primitive
 *  shares — variants compose, caller overrides win). */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
