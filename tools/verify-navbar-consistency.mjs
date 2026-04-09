#!/usr/bin/env node

import { readFileSync } from "node:fs";

const pages = ["index.html", "about.html", "contact.html", "demo.html"];
const sharedCss = readFileSync(new URL("../css/navbar-consistency.css", import.meta.url), "utf8");

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

for (const page of pages) {
  const html = readFileSync(new URL(`../${page}`, import.meta.url), "utf8");
  assert(html.includes("site-navbar"), `${page} is missing the shared navbar class`);
  assert(html.includes("./css/navbar-consistency.css"), `${page} is missing the shared navbar stylesheet`);
  assert(html.includes('href="./about.html"'), `${page} is missing the About link`);
  assert(html.includes('href="./contact.html"'), `${page} is missing the Contact link`);
  assert(html.includes('href="./demo.html"'), `${page} is missing the ALGO link`);
}

const indexHtml = readFileSync(new URL("../index.html", import.meta.url), "utf8");
assert(indexHtml.includes('id="langBtn"'), "index.html is missing the language switch control");
assert(indexHtml.includes("function toggleLanguage()"), "index.html is missing the language toggle behavior");
assert(sharedCss.includes("min-width: 82px"), "Stable language switch width rule is missing");

console.log("Navbar consistency verification passed.");
