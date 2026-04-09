#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");

function run(command, args) {
  return execFileSync(command, args, {
    cwd: repoRoot,
    encoding: "utf8",
  }).trim();
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function collectExcludedPairs() {
  const output = run("python3", [
    "-c",
    `
from pathlib import Path
root = Path("posts")
for md in sorted(root.rglob("*.md")):
    html = md.with_suffix(".html")
    if html.exists():
        print(md.with_suffix(""))
`.trim(),
  ]);

  if (!output) {
    return [];
  }

  return output.split("\n").filter(Boolean);
}

const changedFiles = run("git", ["diff", "--name-only"])
  .split("\n")
  .map((line) => line.trim())
  .filter(Boolean);

const excludedPairs = collectExcludedPairs();
const touchedExcludedFiles = [];

for (const pair of excludedPairs) {
  const mdPath = `${pair}.md`;
  const htmlPath = `${pair}.html`;
  if (changedFiles.includes(mdPath)) {
    touchedExcludedFiles.push(mdPath);
  }
  if (changedFiles.includes(htmlPath)) {
    touchedExcludedFiles.push(htmlPath);
  }
}

assert(
  touchedExcludedFiles.length === 0,
  `Excluded same-named Markdown/HTML files were modified: ${touchedExcludedFiles.join(", ")}`
);

const requiredMarkers = [
  ["contact.html", "content-panel content-panel--narrow content-prose"],
  ["demo.html", "content-link-list"],
  ["about_old.html", "content-inline-actions"],
  ["profile.html", "content-panel content-panel--center content-panel--narrow content-prose profile-card"],
  ["css/content-polish.css", ".content-panel"],
];

for (const [filePath, marker] of requiredMarkers) {
  const absolutePath = path.join(repoRoot, filePath);
  assert(existsSync(absolutePath), `Expected file is missing: ${filePath}`);
  const content = readFileSync(absolutePath, "utf8");
  assert(content.includes(marker), `Expected marker not found in ${filePath}: ${marker}`);
}

const report = [
  "Content presentation verification passed.",
  `Changed files checked: ${changedFiles.length}`,
  `Excluded same-named pairs checked: ${excludedPairs.length}`,
  "Polished independent HTML pages: contact.html, demo.html, about_old.html, profile.html",
];

writeFileSync(
  path.join(repoRoot, "docs/content-presentation-polish-verification.txt"),
  `${report.join("\n")}\n`,
  "utf8"
);

console.log(report.join("\n"));
