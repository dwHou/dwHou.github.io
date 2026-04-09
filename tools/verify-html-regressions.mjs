import fs from "fs";

function read(file) {
  return fs.readFileSync(file, "utf8");
}

const example = read("posts/2021-阅读/Example.html");
if (example.includes("standalone-posts-polish") || example.includes("posts-standalone-polish.css")) {
  throw new Error("Example.html should not use standalone article polish.");
}

const pytorchWithStyle = read("posts/2020-2-深度学习框架-PyTorch常用代码段/PyTorch常用代码段withstyle.html");
if (
  pytorchWithStyle.includes("standalone-posts-polish") ||
  pytorchWithStyle.includes("posts-standalone-polish.css")
) {
  throw new Error("PyTorch常用代码段withstyle.html should not use standalone article polish.");
}

const polishCss = read("css/posts-standalone-polish.css");
for (const selector of [
  "body.standalone-posts-polish #content li",
  "body.standalone-posts-polish #write li",
]) {
  if (!polishCss.includes(selector)) {
    throw new Error(`Expected navbar-safe scoped selector missing: ${selector}`);
  }
}
if (polishCss.includes("body.standalone-posts-polish li + li")) {
  throw new Error("Navbar-unsafe global li selector still exists in posts polish CSS.");
}

const transformer = read("posts/2021-3-DL理论改进/2021-3-Transformer/index.html");
for (const expected of [
  "../../../bootstrap/css/bootstrap.min.css",
  "../../../css/default.css",
  "../../../comments/inlineDisqussions.css",
  "../../../highlight/styles/github.css",
  "../../../about.html",
  "../../../contact.html",
  "../../../demo.html",
  "../../../bootstrap/js/bootstrap.min.js",
]) {
  if (!transformer.includes(expected)) {
    throw new Error(`Transformer page is missing corrected relative path: ${expected}`);
  }
}
for (const broken of [
  'href="../../bootstrap/css/bootstrap.min.css"',
  'href="../../css/default.css"',
  'href="../../comments/inlineDisqussions.css"',
  'href="../../highlight/styles/github.css"',
  'href="../../about.html"',
  'href="../../contact.html"',
  'href="../../demo.html"',
  'src="../../bootstrap/js/bootstrap.min.js"',
]) {
  if (transformer.includes(broken)) {
    throw new Error(`Transformer page still contains broken relative path: ${broken}`);
  }
}

const homepage = read("index.html");
if (!homepage.includes('id="clustrmaps-slot"')) {
  throw new Error("Homepage is missing clustrmaps slot anchor.");
}
if (!homepage.includes("(parent || document.body).appendChild(script)")) {
  throw new Error("Homepage loadScript helper is missing parent-aware insertion.");
}
if (!homepage.includes("clustrmapsContainer || document.getElementById('clustrmaps-slot')")) {
  throw new Error("Homepage ClustrMaps loader is not anchored to the intended slot.");
}

console.log("HTML regression verification passed.");
