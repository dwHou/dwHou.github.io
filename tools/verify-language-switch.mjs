#!/usr/bin/env node

import { readFileSync } from "node:fs";

const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

assert(html.includes('id="langBtn"'), "Language switch button is missing");
assert(html.includes('id="langZh"'), "Chinese language label is missing");
assert(html.includes('id="langEn"'), "English language label is missing");
assert(html.includes("function updateLanguageSwitchState()"), "Language switch state updater is missing");
assert(html.includes("updateLanguageSwitchState();"), "Language switch state updater is not invoked");
assert(html.includes('class="lang-switch-separator">/</span>'), "Language switch separator is missing");
assert(html.includes('./css/navbar-consistency.css'), "Shared navbar consistency stylesheet is missing");
assert(html.includes('href="./about.html"'), "About navbar link changed unexpectedly");
assert(html.includes('href="./contact.html"'), "Contact navbar link changed unexpectedly");
assert(!html.includes("transform: scale(1.05)"), "Old exaggerated hover motion is still present");

console.log("Language switch verification passed.");
