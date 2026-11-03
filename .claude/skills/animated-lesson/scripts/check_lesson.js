#!/usr/bin/env node
// One-pass render check for a lesson page.
// Wraps the fragment the way the Artifact host does, renders desktop (light) and phone (dark),
// and reports console errors, page errors and horizontal overflow. Saves two full-page screenshots.
//
//   node check_lesson.js <lesson.html> [out-dir]
//
// Needs Playwright with a Chromium build. If require('playwright') fails, install it in a scratch
// folder (npm i playwright && npx playwright install chromium) or skip the check.
const fs = require('fs');
const path = require('path');

const src = process.argv[2];
const outDir = path.resolve(process.argv[3] || path.dirname(src || '.'));
if (!src) { console.error('usage: node check_lesson.js <lesson.html> [out-dir]'); process.exit(2); }

let chromium;
try { ({ chromium } = require('playwright')); }
catch (e) { console.error('Playwright not found: skip the visual check, or install it in a scratch folder.'); process.exit(3); }

const body = fs.readFileSync(src, 'utf8');
const wrapped = '<!doctype html><html><head><meta charset="utf-8">' +
  '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">' +
  '<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top);padding-bottom:env(safe-area-inset-bottom)}' +
  'body{margin:0;font:14px system-ui}img{max-width:100%}[hidden]{display:none!important}</style>' +
  '</head><body>' + body + '</body></html>';
fs.mkdirSync(outDir, { recursive: true });
const base = path.basename(src, path.extname(src));
const tmp = path.join(outDir, `${base}.preview.html`);
fs.writeFileSync(tmp, wrapped);

(async () => {
  const browser = await chromium.launch();
  const problems = [];
  const runs = [
    { name: 'desktop-light', opts: { viewport: { width: 1200, height: 900 }, colorScheme: 'light' } },
    { name: 'phone-dark', opts: { viewport: { width: 390, height: 844 }, colorScheme: 'dark' } }
  ];
  for (const r of runs) {
    const page = await browser.newPage(r.opts);
    page.on('console', m => { if (m.type() === 'error') problems.push(`${r.name} console: ${m.text()}`); });
    page.on('pageerror', e => problems.push(`${r.name} page error: ${e.message}`));
    await page.goto('file://' + tmp);
    await page.waitForTimeout(1200);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    if (overflow > 0) problems.push(`${r.name}: page scrolls sideways by ${overflow}px`);
    const shot = path.join(outDir, `${base}.${r.name}.png`);
    await page.screenshot({ path: shot, fullPage: true });
    console.log(`screenshot: ${shot}`);
    await page.close();
  }
  await browser.close();
  fs.unlinkSync(tmp);
  if (problems.length) { console.log('PROBLEMS:\n  ' + problems.join('\n  ')); process.exit(1); }
  console.log('OK: no console errors, no horizontal overflow.');
})();
