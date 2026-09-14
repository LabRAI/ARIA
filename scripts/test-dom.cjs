const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");
const { JSDOM, VirtualConsole } = require(process.env.JSDOM_PATH || "jsdom");
const root = path.resolve(__dirname, "..");
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on("jsdomError", (e) => errors.push(e.message));
const dom = new JSDOM('<div id="root"></div>', {
  url: "http://localhost/",
  runScripts: "outside-only",
  pretendToBeVisual: true,
  virtualConsole,
});
const w = dom.window;
w.scrollTo = () => {};
w.fetch = async (address) => ({
  ok: true,
  json: async () =>
    JSON.parse(
      fs.readFileSync(
        path.join(root, "public", new URL(address, w.location).pathname),
        "utf8",
      ),
    ),
});
const script = fs
  .readdirSync(path.join(root, "dist/assets"))
  .find((f) => f.endsWith(".js"));
w.eval(fs.readFileSync(path.join(root, "dist/assets", script), "utf8"));
const tick = () => new Promise((r) => setTimeout(r, 50));
const click = async (element) => {
  assert(element);
  element.click();
  await tick();
};
const byText = (selector, value) =>
  [...w.document.querySelectorAll(selector)].find((e) =>
    e.textContent.includes(value),
  );
(async () => {
  await tick();
  await tick();
  assert.equal(w.document.querySelectorAll(".chart-row").length, 6);
  assert.equal(w.document.querySelectorAll(".case-card").length, 6);
  const manifest = JSON.parse(
    fs.readFileSync(path.join(root, "public/data/manifest.json"), "utf8"),
  );
  for (let i = 0; i < manifest.cases.length; i++) {
    const entry = manifest.cases[i];
    const data = JSON.parse(
      fs.readFileSync(
        path.join(root, `public/data/${entry.benchmark}.json`),
        "utf8",
      ),
    );
    await click(w.document.querySelectorAll("nav button")[i]);
    await tick();
    assert.equal(w.document.querySelector("h1").textContent, data.review.title);
    assert.equal(w.document.querySelectorAll(".run-summary .badge").length, 3);
    await click(byText(".tabs button", "Trajectory"));
    assert.equal(
      w.document.querySelectorAll(".timeline button").length,
      data.runs.failed.display_trajectory.events.length,
    );
    await click(w.document.querySelector("input[type=checkbox]"));
    assert.equal(
      w.document.querySelectorAll(".event").length,
      data.runs.failed.display_trajectory.events.length,
    );
    const input = w.document.querySelector("input[type=search]");
    Object.getOwnPropertyDescriptor(
      w.HTMLInputElement.prototype,
      "value",
    ).set.call(input, "no-match-82746");
    input.dispatchEvent(new w.Event("input", { bubbles: true }));
    await tick();
    assert.equal(w.document.querySelectorAll(".event").length, 0);
    const select = w.document.querySelector("select");
    for (const mode of ["control", "repaired"]) {
      select.value = mode;
      select.dispatchEvent(new w.Event("change", { bubbles: true }));
      await tick();
      assert.equal(
        w.document.querySelectorAll(".timeline button").length,
        data.runs[mode].display_trajectory.events.length,
      );
      await click(w.document.querySelector(".timeline button"));
      assert(w.document.querySelector(".event[open]"));
    }
    await click(byText(".tabs button", "Verification"));
    assert(
      w.document.body.textContent.includes(data.review.verification_summary),
    );
    assert(w.document.body.textContent.includes(data.review.control_note));
    assert.equal(
      w.document.querySelectorAll(".evaluation-grid .panel").length,
      3,
    );
  }
  await click(w.document.querySelector(".nav-overview"));
  assert.equal(w.document.querySelectorAll(".chart-row").length, 6);
  assert.deepEqual(errors, []);
  console.log(
    "PASS: overview, all six cases, three trajectory modes, full-event counts, search, event selection, verification and navigation.",
  );
  w.close();
})().catch((e) => {
  console.error(e);
  w.close();
  process.exitCode = 1;
});
