import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const root = process.cwd();

function read(relativePath) {
  return readFileSync(resolve(root, relativePath), "utf8");
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function countMatches(source, pattern) {
  const matches = source.match(pattern);
  return matches ? matches.length : 0;
}

const contact = read("contact.html");
const demo = read("demo.html");
const about = read("about.html");
const index = read("index.html");

assert(
  contact.includes('href="#" onclick="emailScramble1.initAnimateBubbleSort();return false;"'),
  "contact.html should keep a valid email unscramble link",
);

assert(
  countMatches(contact, /jquery(?:-[\d.]+)?(?:\.min)?\.js/gi) === 1,
  "contact.html should only load jQuery once",
);

assert(
  countMatches(demo, /jquery(?:-[\d.]+)?(?:\.min)?\.js/gi) === 1,
  "demo.html should only load jQuery once",
);

for (const [file, html] of [
  ["about.html", about],
  ["contact.html", contact],
  ["demo.html", demo],
]) {
  assert(!html.includes('href="//netdna.bootstrapcdn.com'), `${file} should not use protocol-relative Font Awesome`);
  assert(!html.includes('href="http://fonts.googleapis.com'), `${file} should not use insecure Google Fonts URLs`);
  assert(!html.includes('src="http://cdn.mathjax.org'), `${file} should not use insecure MathJax URLs`);
}

assert(index.includes('id="langBtn"'), "index.html should preserve the bilingual language toggle");
assert(about.includes('profile/style/style.css'), "about.html should preserve the mature profile theme");

console.log("Static site verification passed.");
