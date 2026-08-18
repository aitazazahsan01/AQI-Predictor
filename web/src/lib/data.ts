/**
 * Reads the published snapshot at build time.
 *
 * The site is statically exported, so this runs once during `next build` and
 * the result is baked into the HTML. Refreshing the data means rebuilding,
 * which is exactly what a push from the pipeline triggers.
 */

import { readFile } from "node:fs/promises";
import path from "node:path";

import type { Snapshot } from "./types";

const SNAPSHOT_PATH = path.join(process.cwd(), "public", "data", "forecast.json");

/** The version this front end was written against. */
export const SUPPORTED_SCHEMA_VERSION = 1;

export async function loadSnapshot(): Promise<Snapshot> {
  const raw = await readFile(SNAPSHOT_PATH, "utf8");
  const snapshot = JSON.parse(raw) as Snapshot;

  if (snapshot.schema_version !== SUPPORTED_SCHEMA_VERSION) {
    // Fail the build rather than shipping a page that renders half its panels.
    throw new Error(
      `forecast.json is schema v${snapshot.schema_version}, but this site expects ` +
        `v${SUPPORTED_SCHEMA_VERSION}. Rebuild it with scripts/export_web_data.py, ` +
        `or update src/lib/types.ts to match.`,
    );
  }

  return snapshot;
}
