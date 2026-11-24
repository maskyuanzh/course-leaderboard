import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const html = readFileSync(new URL("../../index.html", import.meta.url), "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length, 1, "Expected one inline application script");

class Element {
  children = [];
  dataset = {};
  classList = { add() {} };
  textContent = "";

  append(...children) { this.children.push(...children); }
  add(child) { this.append(child); }
  addEventListener() {}
  replaceChildren() { this.children = []; }
  insertRow() {
    const row = new Element();
    this.append(row);
    return row;
  }
}

const elements = new Map();
const context = vm.createContext({
  assert,
  Option: class {},
  document: {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new Element());
      return elements.get(id);
    },
    createElement: () => new Element(),
    querySelectorAll: () => [],
  },
});

vm.runInContext(scripts[0][1], context, { filename: "index.html" });

await vm.runInContext(String.raw`
  (async () => {
    // Speedrun's batched Muon log has a memory summary after its final evaluation.
    const speedrun = [
      "step:1375/1393 val_loss:3.2820 train_time:177070ms step_avg:129.72ms",
      "step:1393/1393 val_loss:3.2785 train_time:179527ms step_avg:129.81ms",
      "peak memory allocated: 37653 MiB reserved: 39236 MiB",
    ].join("\n");
    const initialScores = JSON.stringify(rankedTeams);
    const historyLength = records.length;
    chooseLog({ name: "speedrun.txt", text: async () => speedrun });
    await checkLog(true);
    assert.match($("file-status").textContent, /Validation Loss 为 3\.2785/);
    assert.equal(JSON.stringify(rankedTeams), initialScores);
    assert.equal(records.length, historyLength);

    const formatted = parseCourseLog(sampleFilename, sample.replaceAll("\n", "\r\n"));
    assert.ok(formatted.title && formatted.description.includes("\n"));
    assert.throws(() => parseCourseLog("unsegmented.txt", sample), /分段命名/);
    assert.throws(() => readLastLoss("Test Loss: 0.22\nTest Loss: NaN", "test[ _]+loss", "Test Loss"), /有限数值/);

    active = true;
    const team = { name: teamName, score: 0.2867, submitted: 2 };
    const otherTeam = { name: "Reference Team", score: 0.2481, submitted: 2 };
    rankedTeams.splice(0, rankedTeams.length, team, otherTeam);
    const initialCount = team.submitted;
    const submit = async (loss) => {
      const log = sample + "\nTest Loss: " + loss + "\npeak memory allocated: 240 MiB";
      chooseLog({ name: sampleFilename, text: async () => log });
      await checkLog();
    };
    await submit(otherTeam.score);
    const leaders = $("standings").children.filter((row) => row.dataset.rank === 1);
    assert.equal(leaders.length, 2, "Equal losses should share first place");
    await submit(0.24805);
    assert.equal($("standings").children[0].children.at(-1).textContent, "0.24805");
    assert.equal(team.title, formatted.title);
    await submit(0.4);
    await submit("NaN");
    assert.equal(team.score, 0.24805, "Worse and invalid runs must preserve the best score");
    assert.equal(team.submitted, initialCount + 3, "Only valid course submissions should count");
  })()
`, context, { filename: "smoke-test" });

console.log("网页校验通过：Speedrun 试读、课程日志、同分排名和最佳分更新。");
