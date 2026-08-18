import { Github } from "lucide-react";

import { formatTimestamp } from "@/lib/format";
import styles from "./SiteHeader.module.css";

const REPO_URL = "https://github.com/aitazazahsan01/AQI-Predictor";

const LINKS = [
  { href: "#forecast", label: "Forecast" },
  { href: "#trend", label: "Trend" },
  { href: "#drivers", label: "Drivers" },
  { href: "#models", label: "Models" },
  { href: "#method", label: "Method" },
];

export function SiteHeader({ generatedAt }: { generatedAt: string }) {
  return (
    <header className={styles.header}>
      <div className={`shell ${styles.inner}`}>
        <a href="#top" className={styles.brand}>
          <span className={styles.mark} aria-hidden="true" />
          Pearls AQI
        </a>

        <nav className={styles.nav} aria-label="Sections">
          {LINKS.map((link) => (
            <a key={link.href} href={link.href} className={styles.link}>
              {link.label}
            </a>
          ))}
        </nav>

        <div className={styles.meta}>
          <span className={styles.stamp}>Updated {formatTimestamp(generatedAt)}</span>
          <a
            href={REPO_URL}
            className={styles.repo}
            target="_blank"
            rel="noreferrer"
            aria-label="Source on GitHub"
          >
            <Github size={16} strokeWidth={2} aria-hidden="true" />
            <span>Source</span>
          </a>
        </div>
      </div>
    </header>
  );
}
