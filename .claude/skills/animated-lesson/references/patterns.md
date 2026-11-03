# Diagram pattern catalog

Every pattern below teaches by motion, and each one fits a particular kind of idea. Pick the pattern that matches what the concept *does*, not the one that looks nicest. All of them assume the helpers in `assets/lesson-template.html`: `$`, `$$`, `el`, `runner`, `tween`, `wait`, `moveAlong`, `restart`, `reduce`.

Two finished lessons in `assets/examples/` contain full working versions:

| Pattern | Full version in |
|---|---|
| 1 Layer stepper | `lesson-concepts-and-comparisons.html` ("Same kitchen, five words") |
| 2 Packet loop | described below |
| 4 Side-by-side race | `lesson-concepts-and-comparisons.html` ("Who decides the next step?") |
| 5 Role ring | `lesson-concepts-and-comparisons.html` ("In a chat window, you are the harness") |
| 7 Swiss cheese | `lesson-layered-defense.html` ("Defense in depth") |
| 8 Mode matrix | `lesson-layered-defense.html` ("Six permission modes") |
| 9 Story in steps | `lesson-layered-defense.html` ("When an allowed command does more…") |
| 12 Self-check sorter | both examples, and the template |

## Contents
1. Layer stepper: things that nest or build on each other
2. Packet loop: a request/response cycle, a pipeline, any loop
3. Before/after cycle: the same actor with and without something
4. Side-by-side race: two strategies on the same task
5. Role ring: the same loop, different actors doing the jobs
6. Zones and cards: what moves where, and what survives
7. Swiss cheese: layered defenses, each catching different things
8. Mode matrix: one setting changes many outcomes
9. Story in steps: one scenario told in 3–5 frames
10. Decision flow: diagnosis by yes/no questions
11. Two lanes, continuous: soft vs hard reliability over many trials
12. Self-check sorter: classify items, with instant feedback
13. Simulator with toggles: many components on and off, with a scorecard

General rules for all of them:
- **Resting state.** Render the finished or most informative frame at load. Start autoplay 2–3 s later, through the same function a button calls.
- **One runner per diagram.** Each button calls `R.start(async alive => …)`, and every `await` is followed by `if (!alive()) return;`.
- **Captions in HTML.** Keep a `<p class="caption">` under the SVG and update it on every beat. The caption does the explaining; the SVG shows.
- **Coordinates.** Pick a viewBox (e.g. `0 0 760 340`), lay boxes out on a simple grid, and keep 14–20px gaps. Keep tokens and moving dots on wires, not on top of text.

---

## 1. Layer stepper

**Use for:** "A contains B contains C", or a picture that builds up one part at a time (prompt → model → tools → agent → workflow → harness).

Each layer is a `<g class="layer" data-step="n">`. Showing step *i* dims layers above *i* and highlights layer *i*. Tabs let the learner jump between steps, and play loops through them.

```js
function show(i) {
  $$('#svg .layer').forEach(g => {
    const s = Number(g.dataset.step);
    g.classList.toggle('dim', s > i);     // .dim { opacity: .16 }
    g.classList.toggle('focus', s === i); // thicker stroke, soft fill
  });
  $('#what').innerHTML = STEPS[i].what;   // definition card under the diagram
}
async function loop(alive) { while (alive()) for (let i = 0; i < STEPS.length; i++) { if (!alive()) return; show(i); await wait(4800); } }
show(STEPS.length - 1); // at rest: everything visible
```

For ambient motion inside a layer (an agent loop spinning), use SVG `<animateMotion>` on a small circle. The template removes these under reduced motion.

## 2. Packet loop

**Use for:** any mechanism where something travels and gets checked: request → policy → tool → world → result → back.

Lay the nodes out on two rows: requests on the top row, results on the bottom row. Define invisible `<path id="p-req1">`-style segments between node centers. A scenario is a list of **beats**. Each beat has a caption and a list of actions, and the beat captions also render as an ordered step list, so the page reads fully at rest.

```js
const SCEN = { tests: [
  { cap: 'The model writes a request. It is only text.', acts: [['msg', 'run: ./mvnw test'], ['move', 'req1']] },
  { cap: 'The harness checks its rules.',               acts: [['gate', 'policy', 'ok']] },
]};
async function act([t, x, y], alive) {
  if (t === 'msg') $('#msg').textContent = x;
  if (t === 'move') { const p = $('#p-' + x); await moveAlong(dot, p, Math.max(500, p.getTotalLength() * 5), false, alive); }
  if (t === 'gate') { const g = $('#g-' + x); g.classList.remove('is-ok', 'is-bad'); g.classList.add('is-' + y); restart(g, 'pulse'); await wait(650); }
}
```

Add a "blocked" scenario that takes a short path back from the policy gate. The contrast with the allowed path is what teaches.

Controls: one button per scenario, plus Pause/Play, **Next step** (runs a single beat) and Replay. After the last beat, wait about 3 s and auto-advance to the next scenario. Under reduced motion, don't auto-advance, or the loop spins with 0-ms waits.

## 3. Before/after cycle

**Use for:** showing that the same actor gives different results depending on the setup (a chef without and with a harness, a request without and with a cache).

Place one central actor and four stations around it. A token walks to each station and back. On arrival, the station flips its text and its good/bad class, and a counter updates ("Mistakes reaching the table: 4"). The cycle alternates modes automatically, and mode buttons let the learner pin one.

```js
async function cycle(mode, alive) {
  for (let i = 0; i < 4; i++) {
    token.setAttribute('opacity', 1);
    await moveAlong(token, paths[i], 650, false, alive); if (!alive()) return;
    renderStation(i, mode);                       // swap text, toggle .bad/.good, restart(g,'pulse')
    await wait(450);
    await moveAlong(token, paths[i], 450, true, alive);
  }
  token.setAttribute('opacity', 0);
}
```

Paths should start at the actor's edge, not its center, so the token never covers the actor's label.

## 4. Side-by-side race

**Use for:** two strategies on the same input. Who decides, how many steps, where each one fails. Examples: workflow vs agent, progress in context vs a progress file, without vs with a check.

Use two lanes, run together with `Promise.all([laneA(alive), laneB(alive)])`. Give each lane a step counter and an outcome box that turns green or red. Add an input toggle ("Easy bug" / "Tricky bug"): the interesting case is the one where the simple strategy fails. Render the finished state of both lanes at rest.

A lane can be:
- an HTML step list that lights each step in turn
- a hub SVG: a center node with 6 tool nodes, where a dot hops center → tool → center for each step and a trace list grows below

Wrap trace text in one `<span>` inside each grid `<li>`. Otherwise `<b>` and text nodes split into separate grid cells.

## 5. Role ring

**Use for:** the same loop of jobs, but who performs each job changes (you vs the harness, a person vs automation).

Place six nodes on an ellipse, and position the token by angle: `x = cx + rx·cos a`, `y = cy + ry·sin a`. Each node has a step label and a "who" line colored by role. Toggling the mode relabels the nodes and recolors the token. A big counter in the middle counts the human's actions (12 vs 2). Run the loop twice: the first round fails and the second passes, which makes the repetition felt.

## 6. Zones and cards

**Use for:** anything that moves between places or survives or vanishes. Examples: context vs memory vs state, cache tiers, message queues.

Draw four labeled zone rectangles. Cards are `<g>` groups (rect + color tick + label) moved with `transform`. Each step declares which card IDs sit in which zone. A tiny layout engine assigns slot coordinates and tweens every card from its old to its new position: a card arriving from its source (a copy from memory) starts at the source's position, and a leaving card fades out.

```js
const STEPS = [ { counter: ['rules', 'task'], drawer: ['note1'], cap: 15, text: '…' }, … ];
const slot = { counter: i => [268 + (i % 3) * 158, 48 + Math.floor(i / 3) * 36], drawer: i => [30, 178 + i * 36] };
// for each card: target = layout[id]; had = pos[id]
//   new → from ORIGIN[id] or off-screen; moved → from had; removed → fade at had
```

Add a capacity gauge (a rect whose width is a percentage, red above 80%) to show a filling limit, and a "wipe" flash for an end-of-session reset. The step buttons are named moments ("Morning", "Compact", "Night"), not numbers.

## 7. Swiss cheese

**Use for:** defense in depth, where several independent layers each stop different failures.

Draw 4–6 vertical slabs in concept colors, decorated with a few "holes" (circles). A policy bar across the top sets all of them. Projectiles fly left to right, one per action. Each action names the layer that stops it (or none). On impact, the slab flashes, the dot turns red and falls, and a log line appears: ✗ action, a colored chip for the layer, and the reason. Keep the log to the last 4 lines.

Order the actions to tell a story: one safe action first, then the strongest restriction (no tool at all), then the permission, guardrail, constraint and environment examples.

## 8. Mode matrix

**Use for:** one setting that changes many outcomes (permission modes, isolation levels, cache policies, consistency levels).

One tab per mode. Rows are 4–6 typical actions, and each row shows a result pill: Runs / Asks / Classifier / Waits / Refused. Switching modes flips the pills one after another (CSS `rotateX` keyframe, about 90 ms stagger), and a "best for" line updates. Take every cell from the official docs; this is where invented details mislead most.

## 9. Story in steps

**Use for:** one concrete incident that shows why two mechanisms are both needed (for example, an allowed command that does more than its name suggests).

Draw a small SVG of 5–7 boxes. A single **Run** button plays 4–5 beats: highlight a box, color it green or red, move one or two dots, and rewrite the caption each time. At rest, show the final frame, with the outcome colored.

## 10. Decision flow

**Use for:** diagnosis — "which lever fixes this?"

Stack the questions vertically on the left, with "yes" arrows going right to answer boxes in concept colors and "no" arrows going down. The learner picks a symptom, guesses the answer, and then a dot walks down the questions to the right "yes" and across to the answer. The verdict shows right or wrong, the reasoning, and an "also helps" note, because real failures usually have a second lever.

Keep question text under about 36 characters at 13px in a 300px box. Put the detail in a muted second line.

## 11. Two lanes, continuous

**Use for:** reliability over many trials, showing soft vs hard: a written rule that is followed most of the time vs a check that runs every time.

Two horizontal roads. The soft lane has a detour path past a note box; the hard lane has a gate across the road. Spawn a packet into each lane every 1.7 s. Soft-lane packets take the detour by a fixed pattern (e.g. `[1,1,0,1]`); a packet that passes the note or gate point turns green. Counters show "tests ran 3 of 4". Skip spawning when `document.hidden`, reset the counters after about 40, and say on the page that the rate is made up for the picture.

## 12. Self-check sorter

**Use for:** the end-of-section self-check: classify 8–10 items into 3–6 categories.

It's included in the template. Make 2–3 of the items deliberately tricky, with an explanation that teaches the nuance ("It's loaded into context, but it's a file the harness loads every session, so it's harness"). Answers can include a third option such as "Rebuildable" when the binary is false.

## 13. Simulator with toggles

**Use for:** a hands-on page where the learner turns components on and off and watches an agent's session change.

- **Layer cards.** Each is a toggle button showing ON/OFF, its kind (soft/hard) and a "fired N×" counter that flashes when that layer acts.
- **A scripted `sim(on, seed)`.** It returns transcript lines and a scorecard. Precompute every random decision from the seed before branching on toggles, so turning one layer on or off doesn't reshuffle the others.
- **A dark terminal panel.** Lines appear with per-type delays (user, say, tool, out, harness, note, final).
- **A scorecard.** Each row is "Enforced", "OK, not enforced" or "Failed". The middle state teaches the difference between luck and a guarantee.
- **Buttons.** Presets ("None", "Soft only", "All"), New run (next seed), Replay and Show result. At rest, show a finished run that has something interesting in it: search seeds for one where the hard layers fire.
- **Honesty.** Say on the page that it's a scripted simulation and that the rates are illustrative.
