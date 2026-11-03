# Teaching guide

How to turn a topic into a sequence of lessons that a smart beginner actually understands, and how to coach them through it.

## Contents
1. Planning a series
2. The lesson sequence
3. Choosing the analogy
4. Concept cards and "don't mix these up"
5. Knowledge checks and evaluating answers
6. Honesty rules
7. Tone

## 1. Planning a series

- **Start with terminology when it's contested.** If the topic's main word has several meanings (a "harness" in wiring, testing, a company name, AI agents), lesson 1 says so in a compact card grid and marks the one this course covers. If the term is new, say roughly when it appeared and that definitions still differ.
- **Group concepts into families.** A roadmap of 18 concepts becomes four lessons of 3–6. Show the family map near the top of each lesson, with the current lesson highlighted and earlier ones marked done.
- **Reorder when it helps, and say so in one line.** "Your roadmap puts this at Level 3; it follows directly from Lesson 2, so it comes now."
- **Keep one thread through the series.** The same analogy world, the same running example (one service, one task, such as "add a $25,000 ACH limit"), and the same tokens and fonts. Later lessons can say "the same trainee chef from Lesson 1" and the learner instantly has the picture.
- **Hands-on pages.** When the learner asks for an example to implement, build a simulator page (pattern 13) and offer to create a real runnable project. If you create one, verify it actually runs, and say what you verified and what you couldn't.

## 2. The lesson sequence

Use this order inside each lesson, and at the level of each concept:

1. **Simple idea.** One sentence a 10-year-old could follow.
2. **Real-world analogy.** Mapped item by item in a small table or diagram.
3. **Tiny example.** One concrete case, small enough to hold in your head.
4. **Technical explanation.** Now introduce the real terms, each defined on first use.
5. **Real engineering example.** In the learner's stack and domain.
6. **Hands-on.** A self-check on the page, and "apply it" snippets for their project.
7. **Check understanding.** The knowledge check, answered in chat.

Put the core insight in a short callout once per lesson, phrased so the learner could repeat it. For example: "The model can't touch anything. It only produces text." or "The model only has context; everything else must be stored outside it."

## 3. Choosing the analogy

- It must map the **mechanism**, not just the vibe. Test it: does every concept in the lesson have a natural counterpart? A kitchen worked for agents because it has equipment (tools), rules on the wall (instructions), a counter (context), a notebook (memory), an order board (state), a head chef tasting (validation), locked rooms (permissions) and head-office rules (policy).
- Say where it breaks when the break is instructive. "Why a kitchen and not a classroom? A student learns and remembers. A model doesn't."
- Give the analogy its own element in the "six fields" card, so the learner can always find it.

## 4. Concept cards and "don't mix these up"

Every concept gets these six fields:

| Field | Content |
|---|---|
| What it is | One or two plain sentences |
| Why we need it | The problem it solves |
| Analogy | In the series' analogy world |
| Simple example | Non-software, everyday |
| Software example | In the learner's domain |
| In your tools | Exactly where it lives in their actual tools: file names, commands, settings |

After the cards, add three "don't mix these up" pairs for the concepts most likely to blur. Each pair gets one sentence that states the difference, and ideally one discriminating question ("A permission says yes or no. A constraint limits something that is allowed.").

## 5. Knowledge checks and evaluating answers

End each lesson with four blocks:

1. **Quick quiz (3–5).** Short recall plus one "why" question. Include one true/false that targets a common misconception ("A detailed enough prompt can stop an agent pushing to main").
2. **Scenario.** A realistic situation from the learner's work, where they must decide something.
3. **Design challenge.** "Write one concrete item for each concept for this change", using the running example.
4. **Explain-back.** Explain it to a colleague, or without software words. This exposes memorized-but-not-understood answers.

When the learner answers:

- **What they understood correctly.** Be specific.
- **What they misunderstood, and why it's wrong.** Correct it plainly. Don't soften an error into "partially right" when it isn't.
- **What they missed.**
- **How an experienced engineer would improve it.** Give a short model answer at senior, staff or principal level when asked.

Then decide whether they're ready for the next lesson. If they skip the checks and ask to continue, continue: it's their course. Mention the open checks once, in a short line, and suggest the single most valuable one to try.

## 6. Honesty rules

- **Verify fast-moving facts against primary sources** before teaching them, and quote only what you read. List sources at the bottom of the page.
- **Hedge plainly** where the field is unsettled ("these boundaries are fuzzy; people use the terms loosely").
- **Label illustrative numbers and simulations on the page itself.** A scorecard or rate that looks like data but isn't, if unlabeled, teaches something false.
- **Separate guaranteed from likely.** Soft mechanisms (instructions, memory) make good behavior likely; hard ones (permissions, hooks, sandboxes, tests) make bad behavior impossible or detectable. Many topics have this split, and naming it is often the most useful idea in the lesson.

## 7. Tone

Write for an intelligent person who is new to the topic. Use short, active sentences and second person. Introduce a term only when you need it, then use it consistently. Don't use em-dash asides, "not X but Y" constructions, scare quotes or stock phrases such as "worth noting". Put the explanations in captions and cards, and keep the chat reply short: the link, the core idea, what's on the page, what was verified, and what's next.
