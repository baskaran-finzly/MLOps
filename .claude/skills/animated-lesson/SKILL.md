---
name: animated-lesson
description: Build animated, interactive single-page HTML lessons that teach a technical topic from beginner level, one concept at a time, with a running real-world analogy, live SVG diagrams the learner can play and step through, a self-check they can click through, and a knowledge check to answer in chat. Use this whenever someone wants to learn, be taught, or be coached on a concept and a visual page would help — "teach me X", "explain X like I'm new", "next lesson", "continue to lesson N", "make this a course", "animated diagrams would be more helpful", a tutorial, explainer, module, workshop or study guide — even if they don't say "artifact" or "HTML". Also use it to continue an existing lesson series in the same look. Prefer it over a plain text explanation whenever the topic has moving parts (a loop, a pipeline, layers, a before/after, a lifecycle) that animation can show.
---

# Animated lesson

Teach one idea at a time on one page, so a beginner can watch the mechanism move instead of reading about it. The output is a single self-contained HTML file: inline CSS and JS, Google Fonts only, light and dark themes, phone-width safe. Where an Artifact tool is available, publish it. Otherwise save the file for the user to open.

These pages were developed while teaching a course on harness engineering for AI coding agents. Two finished lessons are in `assets/examples/` and show the target quality. Open one before building your first page.

## What a good lesson does

- **Simple first.** Every lesson follows this sequence: simple idea → real-world analogy → tiny example → technical explanation → real engineering example → hands-on → check understanding. Never use a term before it has been explained.
- **One family of concepts per lesson.** Three to six closely related ideas at most. If the request covers more, split it into a series and say so at the top ("this lesson covers the first family; the others come in Lessons 5–7").
- **Animation carries the mechanism.** Use 3–5 interactive diagrams, each tied to exactly one idea. One of them is the centerpiece. Decoration that teaches nothing doesn't belong.
- **One analogy throughout.** Pick one everyday world (a kitchen, an airport, a library) and map every concept onto it. Keep it across the whole series, so the learner builds on what they already pictured. Choose the analogy that maps accurately, and say where it breaks down if that teaches something. For example: "a classroom suggests the student remembers; a model doesn't".
- **The learner's own domain.** Every concept gets a concrete example from the learner's real work. Use what you know about them: their stack, industry and projects. A Java payments engineer gets `PaymentService`, ACH limits and Kafka, not `foo()`.
- **Honest.** Say when a term has several meanings, when definitions differ, when something depends on a version, and when a number is illustrative. Never present invented figures as measurements.

## Workflow

### 1. Understand the request

Work out the topic, the learner's level, and their domain. Then decide whether this is a new series or the next lesson of an existing one. Conversation history usually answers all four; ask only for what you can't infer. If the learner skipped earlier knowledge checks, continue anyway. Mention once, briefly, that the checks are still open, and don't nag.

### 2. Research before teaching anything recent

For anything that changes fast (AI tooling, cloud services, framework versions, new terminology), verify against primary sources before writing: official docs, the original blog post, the spec. Quote exact phrases where they help, and only quotes you actually read. Add a Sources list at the bottom of the page. Keep every fact you're unsure of out of the page, or hedge it plainly ("roughly when the term became popular").

### 3. Plan the page

Before writing code, sketch the plan in a few lines:

- **Sections**, in order. Usual shape:
  1. hero: eyebrow with series and lesson number, title, one-paragraph lede
  2. optional map of where this lesson sits in the series
  3. the centerpiece animation
  4. a template card for each concept (see below)
  5. "Don't mix these up" pairs
  6. one or two supporting animations
  7. a clickable self-check
  8. "Apply it" with real snippets from the learner's project
  9. a one-sentence summary
  10. a knowledge check
  11. sources
- **Concept template.** For each concept, give six short fields: what it is, why we need it, analogy, simple example, software example, and how it looks in the learner's actual tools. Use tabs, so only one concept is open at a time.
- **Diagrams.** Pick them from `references/patterns.md`, which has a catalog and code. Match each pattern to what the idea *does*:
  - a loop: packet along a path
  - layers: a stepper with dimmed layers, or Swiss-cheese slabs
  - before and after: a toggle cycle
  - two strategies: a side-by-side race
  - something that survives or is lost: a zones layout
  - a choice that changes outcomes: a mode matrix
  - diagnosis: a decision flow
- **Knowledge check.** End each lesson (or each level of a course) with:
  - a quick quiz (3–5)
  - a realistic scenario
  - a design challenge
  - an explain-back test ("explain it to a colleague without software words")

  Tell the learner to answer in chat, since you evaluate the answers there. When they answer, say what they got right, what they misunderstood and what they missed, and how an experienced engineer would improve it.

`references/pedagogy.md` covers analogies, sequencing, the knowledge check and the tone in more depth. Read it when starting a new series.

### 4. Build from the template

Copy `assets/lesson-template.html` and fill it in. The template already provides:

- color and type tokens for both themes
- the side gutter with safe-area padding
- shared components: bench, caption, buttons, tabs, concept grid, pairs, self-check sorter, quiz, summary, sources
- the animation helpers: `runner`, `tween`, `wait`, `restart` and `el`, with reduced-motion handling

Keep its token names. For a series, reuse the same tokens and fonts in every lesson, so the pages read as one course. For a new series, you may choose a new palette and fonts. Change the token values; don't scatter literal colors.

Rules that keep the pages working. Each one exists because a lesson broke without it:

- **Complete at rest.** The first frame must already teach something: a filled diagram, a visible log, a caption. Autoplay can start a second or two later from that visible state. Never leave content at `opacity: 0` waiting for a trigger; dimmed future steps at about 0.16–0.22 opacity are fine.
- **Every animation can be paused or stepped**, and restarting cancels the old run. Use `runner()`: each run checks `alive()` after every await, so clicking another button never leaves two animations fighting over one element.
- **Reduced motion.** Under `prefers-reduced-motion`, no autoplay, tweens jump to their end, and `animateMotion` elements are removed. Buttons still work.
- **SVG diagrams** live in `.scroll` with `min-width` around 620–660px, so they scroll sideways on phones instead of shrinking text to nothing. Put long explanations in HTML captions, not in SVG text.
- **SVG text overflows silently.** Keep labels short, about 22 characters at 11–13px in a 150px box. Wrap longer text in HTML.
- **A flex or grid row with inline `<code>`** splits into separate lines unless the text is wrapped in one `<span>` or `<p>`. Style `.quiz > li`, not `.quiz li`, or the nested quiz items break the same way.
- **Colors come only from tokens**, including inside SVG (`fill: var(--accent)` in CSS, not in attributes), and `body` sets its own background.
- **Illustrative numbers are labeled as illustrative** on the page itself ("a made-up rate for the picture, not a measurement"). A seeded random generator is fine for simulations, as long as the page says so.
- **Copy is plain.** Use short active sentences. Don't use em-dash asides, "not X but Y" framing, or stock phrases. Buttons say what they do ("Run both", "Check my answers").

### 5. Check once, then publish

Run `node scripts/check_lesson.js <file.html> <out-dir>` if Node and Playwright are available. It wraps the page the way the Artifact host does, renders desktop light and phone dark, and prints console errors and horizontal overflow. It also saves two screenshots: look at them once. Fix what's broken in one pass, then publish. Don't loop on polish; the learner reviews the live page.

To publish, call the Artifact tool's `quickstart` with `intent: "other"` (and `design_systems: false`) the first time in a session. Then publish the file with a short `icon` (e.g. `book`) and a one-sentence `description`. Use a title that names the lesson ("Kafka Basics Lesson 3"), not a sentence. For later fixes to the same lesson, republish the same file path with `edits`. Without an Artifact tool, save the HTML where the user can open it and give the path.

### 6. Reply in chat

Keep the reply short and scannable:
- the link
- the lesson's core idea in one or two sentences
- a numbered list of what's on the page, one line per section
- facts you verified, with sources
- what comes next
- one specific knowledge-check question to start with

Don't repeat the whole page in chat.

## Files in this skill

- `assets/lesson-template.html`: the starting point. Tokens, components, helpers and placeholder sections.
- `references/patterns.md`: the diagram pattern catalog, with working code for each.
- `references/pedagogy.md`: how to sequence a course, choose analogies, write knowledge checks and evaluate answers.
- `assets/examples/lesson-concepts-and-comparisons.html`: a finished lesson with a word-by-word stepper, a workflow vs agent race, a "you are the harness" loop, a prompt-vs-enforcement grid and a sorter.
- `assets/examples/lesson-layered-defense.html`: a finished lesson with a Swiss-cheese defense animation, concept tabs, a mode matrix, a sandbox story and "apply it" snippets.
- `scripts/check_lesson.js`: the one-pass render check.
