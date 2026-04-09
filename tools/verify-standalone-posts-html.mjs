import fs from "fs";
import path from "path";

const TYPOra_POLISH = new Set([
  "posts/2021-阅读/2021书单.html",
  "posts/2022-5-几种学习/index.html",
  "posts/2022-9-colorspace/index.html",
]);

const SKIP_PATTERNS = [
  /\/2021-3-Demo\//,
  /\/2020-8-CompareKit\//,
  /\/2022-3-感受野工具\//,
];

const MARKERS = ["posts-standalone-polish.css", "standalone-posts-polish"];

function walk(dir, files = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(fullPath, files);
    } else if (entry.isFile() && fullPath.endsWith(".html")) {
      files.push(fullPath.replace(/\\/g, "/"));
    }
  }
  return files;
}

function hasMarker(text) {
  return MARKERS.some((marker) => text.includes(marker));
}

function isExcluded(file) {
  const mdPath = path.join(
    path.dirname(file),
    `${path.basename(file, ".html")}.md`,
  );
  return fs.existsSync(mdPath);
}

function isSharedTemplateArticle(file, text) {
  return (
    text.includes("default.css") &&
    text.includes('id="content"') &&
    !SKIP_PATTERNS.some((pattern) => pattern.test(file))
  );
}

function isPolished(file, text) {
  return TYPOra_POLISH.has(file) || isSharedTemplateArticle(file, text);
}

const htmlFiles = walk("posts").sort();
const excluded = [];
const polished = [];
const intentionallyUntouched = [];

for (const file of htmlFiles) {
  const text = fs.readFileSync(file, "utf8");
  if (isExcluded(file)) {
    excluded.push(file);
    if (hasMarker(text)) {
      throw new Error(`Excluded same-name md/html page was modified: ${file}`);
    }
    continue;
  }

  if (isPolished(file, text)) {
    polished.push(file);
    for (const marker of MARKERS) {
      if (!text.includes(marker)) {
        throw new Error(`Polished page is missing marker "${marker}": ${file}`);
      }
    }
    continue;
  }

  intentionallyUntouched.push(file);
  if (hasMarker(text)) {
    throw new Error(`Intentionally untouched page has polish marker: ${file}`);
  }
}

if (excluded.length !== 42) {
  throw new Error(`Expected 42 excluded md/html pairs, got ${excluded.length}`);
}

if (polished.length !== 34) {
  throw new Error(`Expected 34 polished standalone pages, got ${polished.length}`);
}

console.log("Standalone posts verification passed.");
console.log(`Excluded md/html pairs: ${excluded.length}`);
console.log(`Polished standalone HTML pages: ${polished.length}`);
console.log(`Intentionally untouched eligible pages: ${intentionallyUntouched.length}`);
