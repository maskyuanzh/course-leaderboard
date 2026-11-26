import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { basename } from "node:path";
import vm from "node:vm";

const html = readFileSync(new URL("../../index.html", import.meta.url), "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length, 1, "Expected one inline application script");
assert.match(html, /href="\.\/rules\.html"/);
assert.doesNotMatch(html, /rules-dialog|course-leaderboard#readme|class="metric"/);

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
  location: { protocol: "file:" },
  capturedLog: process.argv[2] && {
    name: basename(process.argv[2]),
    contents: readFileSync(process.argv[2], "utf8"),
  },
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
    assert.ok(formatted.title && formatted.description.includes("\n") && formatted.steps === 3);
    assert.throws(() => parseCourseLog("unsegmented.txt", sample), /分段命名/);
    assert.throws(() => readLastLoss("Test Loss: 0.22\nTest Loss: NaN", "test[ _]+loss", "Test Loss"), /有限数值/);
    assert.throws(() => parseCourseLog(sampleFilename, sample.replace(/^step:2.*\n/m, "")), /编号/);
    assert.throws(() => parseCourseLog(sampleFilename, sample.replace("Exit Code: 0", "Exit Code: 1")), /尚未成功/);
    assert.throws(() => parseCourseLog(sampleFilename, sample.replace('"steps":3', '"steps":4')), /实际训练步数/);
    assert.throws(() => parseCourseLog(sampleFilename, sample.replace('{"path":', '{"unused":')), /源码记录/);

    const teacherRows = [
      { name: "Third Team", title: "Baseline", score: 0.4 },
      { name: teamName, title: "Reviewed", score: 0.24805 },
      { name: "Reference Team", title: "Fixed batch", score: 0.24805 },
    ];
    assert.throws(() => parsePublishedResults([...teacherRows, teacherRows[0]]), /Invalid results row/);
    rankedTeams = parsePublishedResults(teacherRows);
    renderBoard();
    const leaders = $("standings").children.filter((row) => row.dataset.rank === 1);
    assert.equal(leaders.length, 2, "Equal losses should share first place");
    assert.equal($("standings").children[0].children.at(-1).textContent, "0.24805");
    const scoredBoard = JSON.stringify(rankedTeams);
    const fakeScore = sample.replace("[Run status]", "Test Loss: 0.001\n[Run status]");
    chooseLog({ name: sampleFilename, text: async () => fakeScore });
    await checkLog();
    assert.equal(records[0].steps, 3);
    assert.equal(records[0].title, formatted.title);
    assert.equal(JSON.stringify(rankedTeams), scoredBoard, "Student logs cannot edit teacher results");
    const directory = "submissions/kernel-studio/run-006";
    assert.equal($("handoff").hidden, false);
    assert.equal($("model-dir").value, directory + "/model");
    const receipt = buildSubmission(checkedLog, directory + "/model/", directory + "/infer.py");
    assert.equal(receipt.log, directory + "/" + sampleFilename);
    assert.equal(receipt.model_dir, directory + "/model");
    assert.equal(receipt.inference_script, directory + "/infer.py");
    assert.throws(() => buildSubmission(checkedLog, directory + "/../model", directory + "/infer.py"), /路径/);
    assert.throws(() => buildSubmission(checkedLog, directory + "/model", directory + "/infer.sh"), /\.py/);
    chooseLog({ name: sampleFilename, text: async () => sample, demo: true });
    await checkLog();
    assert.equal($("handoff").hidden, true);
    assert.equal(checkedLog, null);
    if (capturedLog) assert.equal(parseCourseLog(capturedLog.name, capturedLog.contents).steps, 3);
  })()
`, context, { filename: "smoke-test" });

console.log("网页校验通过：Speedrun 试读、完整日志预检、教师成绩与同分排名。");
