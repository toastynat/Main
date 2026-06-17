// Snip — auto-cut silences & caption videos, 100% in the browser.
//
// The whole pipeline runs on the user's device:
//   1. Read the chosen video's audio (Web Audio API)
//   2. Find the silent gaps and work out which parts to keep
//   3. Transcribe the audio with Whisper (transformers.js) for captions
//   4. Re-cut the video with ffmpeg.wasm and hand back a download
//
// Nothing is ever uploaded. The two heavy libraries are loaded from a CDN.

import { FFmpeg } from "https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.10/dist/esm/index.js";
import { fetchFile, toBlobURL } from "https://cdn.jsdelivr.net/npm/@ffmpeg/util@0.12.1/dist/esm/index.js";
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2";

// Let transformers.js download models from the Hugging Face hub.
env.allowLocalModels = false;

// ---- tiny DOM helpers --------------------------------------------------
const $ = (id) => document.getElementById(id);
const show = (el) => el.classList.remove("hidden");
const hide = (el) => el.classList.add("hidden");

const els = {
  picker: $("picker"), file: $("file"), pickBtn: $("pickBtn"),
  setup: $("setup"), preview: $("preview"), fileMeta: $("fileMeta"),
  doCaptions: $("doCaptions"), doCut: $("doCut"),
  thresh: $("thresh"), threshLabel: $("threshLabel"),
  gap: $("gap"), gapLabel: $("gapLabel"), model: $("model"),
  runBtn: $("runBtn"), resetBtn: $("resetBtn"),
  progress: $("progress"), stage: $("stage"), barFill: $("barFill"), detail: $("detail"),
  results: $("results"), resultVideo: $("resultVideo"),
  dlVideo: $("dlVideo"), dlSrt: $("dlSrt"), summary: $("summary"),
  transcript: $("transcript"), againBtn: $("againBtn"),
  error: $("error"),
};

let currentFile = null;
let ffmpeg = null;        // lazily created ffmpeg.wasm instance
let transcriber = null;   // lazily created Whisper pipeline
let loadedModelId = null;

// ---- progress UI -------------------------------------------------------
function setStage(text) { els.stage.textContent = text; }
function setDetail(text) { els.detail.textContent = text || ""; }
function setProgress(frac) { els.barFill.style.width = `${Math.round(Math.max(0, Math.min(1, frac)) * 100)}%`; }

function fail(message, err) {
  console.error(err || message);
  hide(els.progress);
  els.error.textContent = message;
  show(els.error);
}

// ---- step 1: pick a file ----------------------------------------------
els.pickBtn.addEventListener("click", () => els.file.click());
els.file.addEventListener("change", () => { if (els.file.files[0]) loadFile(els.file.files[0]); });

// drag & drop
els.picker.addEventListener("dragover", (e) => { e.preventDefault(); els.picker.classList.add("dragover"); });
els.picker.addEventListener("dragleave", () => els.picker.classList.remove("dragover"));
els.picker.addEventListener("drop", (e) => {
  e.preventDefault();
  els.picker.classList.remove("dragover");
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("video/")) loadFile(f);
});

function loadFile(file) {
  currentFile = file;
  hide(els.error);
  els.preview.src = URL.createObjectURL(file);
  els.fileMeta.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB`;
  hide(els.picker);
  show(els.setup);
}

// live labels for the sliders
els.thresh.addEventListener("input", () => els.threshLabel.textContent = `(${els.thresh.value} dB)`);
els.gap.addEventListener("input", () => els.gapLabel.textContent = `(${(els.gap.value / 1000).toFixed(1)} s)`);

els.resetBtn.addEventListener("click", resetToStart);
els.againBtn.addEventListener("click", resetToStart);
function resetToStart() {
  hide(els.setup); hide(els.results); hide(els.progress); hide(els.error);
  els.file.value = "";
  currentFile = null;
  show(els.picker);
}

// ---- step 2: run -------------------------------------------------------
els.runBtn.addEventListener("click", run);

async function run() {
  try {
    hide(els.setup); hide(els.error); show(els.progress);
    setProgress(0);

    const wantCaptions = els.doCaptions.checked;
    const wantCut = els.doCut.checked;
    if (!wantCaptions && !wantCut) {
      fail("Pick at least one thing to do — captions, cutting, or both.");
      return;
    }

    // --- decode the audio ---
    setStage("Reading the audio…");
    setProgress(0.05);
    const audioBuffer = await decodeAudio(currentFile);

    // --- find which parts to keep ---
    let segments = [{ start: 0, end: audioBuffer.duration }];
    if (wantCut) {
      setStage("Finding the quiet bits…");
      setProgress(0.15);
      segments = detectKeepSegments(audioBuffer, {
        thresholdDb: Number(els.thresh.value),
        minSilenceMs: Number(els.gap.value),
      });
      if (segments.length === 0) segments = [{ start: 0, end: audioBuffer.duration }];
    }

    // --- captions ---
    let words = [];
    if (wantCaptions) {
      setStage("Listening for words… (first run downloads the model)");
      setProgress(0.25);
      const mono16k = await resampleMono16k(audioBuffer);
      words = await transcribe(mono16k, els.model.value);
    }

    // build SRT against the *trimmed* timeline
    const srt = wantCaptions ? buildSrt(words, segments) : "";

    // --- re-cut the video ---
    setStage("Putting your video back together…");
    setProgress(0.7);
    let outBlob;
    if (wantCut && !(segments.length === 1 && segments[0].start === 0 && Math.abs(segments[0].end - audioBuffer.duration) < 0.01)) {
      outBlob = await cutVideo(currentFile, segments);
    } else {
      // nothing to trim — just hand back the original
      outBlob = currentFile;
    }

    showResults(outBlob, srt, words, audioBuffer.duration, segments);
  } catch (err) {
    fail("Something went wrong: " + (err?.message || err) + ". Try a shorter clip, or see the console for details.", err);
  }
}

// ---- audio decoding ----------------------------------------------------
async function decodeAudio(file) {
  const arrayBuf = await file.arrayBuffer();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  try {
    // decodeAudioData wants its own copy of the buffer
    return await ctx.decodeAudioData(arrayBuf.slice(0));
  } finally {
    ctx.close();
  }
}

// Resample to 16 kHz mono Float32 — the format Whisper expects.
function resampleMono16k(audioBuffer) {
  const targetRate = 16000;
  const length = Math.ceil(audioBuffer.duration * targetRate);
  const offline = new OfflineAudioContext(1, length, targetRate);
  const src = offline.createBufferSource();
  src.buffer = audioBuffer;
  src.connect(offline.destination);
  src.start();
  // OfflineAudioContext.startRendering returns a promise, but we want sync-ish
  // flow; callers await it.
  return offline.startRendering().then((rendered) => rendered.getChannelData(0));
}

// ---- silence detection -------------------------------------------------
// Walk the loudest channel in ~20 ms windows, mark windows below the
// threshold as "silent", then keep everything that isn't a long-enough gap.
function detectKeepSegments(audioBuffer, { thresholdDb, minSilenceMs }) {
  const rate = audioBuffer.sampleRate;
  const data = audioBuffer.getChannelData(0);
  const win = Math.floor(rate * 0.02) || 1;     // 20 ms windows
  const threshold = Math.pow(10, thresholdDb / 20); // dB -> linear amplitude
  const minSilenceWins = Math.ceil((minSilenceMs / 1000) / 0.02);
  const padding = 0.05; // keep 50 ms around speech so words aren't clipped

  // classify each window
  const loud = [];
  for (let i = 0; i < data.length; i += win) {
    let sum = 0;
    const end = Math.min(i + win, data.length);
    for (let j = i; j < end; j++) sum += data[j] * data[j];
    const rms = Math.sqrt(sum / (end - i));
    loud.push(rms >= threshold);
  }

  // collapse short silences (treat them as still "loud" so we don't over-cut)
  let silenceRun = 0;
  for (let i = 0; i < loud.length; i++) {
    if (!loud[i]) {
      silenceRun++;
    } else {
      if (silenceRun > 0 && silenceRun < minSilenceWins) {
        for (let k = i - silenceRun; k < i; k++) loud[k] = true;
      }
      silenceRun = 0;
    }
  }

  // turn runs of "loud" windows into time segments
  const segments = [];
  let segStart = null;
  for (let i = 0; i < loud.length; i++) {
    const t = i * 0.02;
    if (loud[i] && segStart === null) segStart = t;
    if (!loud[i] && segStart !== null) {
      segments.push({ start: Math.max(0, segStart - padding), end: t + padding });
      segStart = null;
    }
  }
  if (segStart !== null) segments.push({ start: Math.max(0, segStart - padding), end: audioBuffer.duration });

  // merge segments that overlap after padding
  const merged = [];
  for (const s of segments) {
    const last = merged[merged.length - 1];
    if (last && s.start <= last.end) last.end = Math.max(last.end, s.end);
    else merged.push({ ...s });
  }
  return merged;
}

// ---- transcription -----------------------------------------------------
async function transcribe(mono16k, modelId) {
  if (!transcriber || loadedModelId !== modelId) {
    transcriber = await pipeline("automatic-speech-recognition", modelId, {
      progress_callback: (p) => {
        if (p.status === "progress" && p.total) {
          setDetail(`Downloading caption model… ${Math.round((p.loaded / p.total) * 100)}%`);
          setProgress(0.25 + 0.2 * (p.loaded / p.total));
        }
      },
    });
    loadedModelId = modelId;
  }
  setDetail("Transcribing…");
  const output = await transcriber(mono16k, {
    return_timestamps: "word",
    chunk_length_s: 30,
    stride_length_s: 5,
  });
  setDetail("");
  // normalise to [{text, start, end}]
  const chunks = output.chunks || [];
  return chunks
    .filter((c) => c.timestamp && c.timestamp[0] != null)
    .map((c) => ({
      text: c.text.trim(),
      start: c.timestamp[0],
      end: c.timestamp[1] ?? c.timestamp[0] + 0.3,
    }));
}

// ---- SRT building ------------------------------------------------------
// Remap each word from the original timeline onto the trimmed timeline,
// dropping words that fell inside a cut, then group words into short cues.
function buildSrt(words, segments) {
  // cumulative kept-time before each segment
  const offsets = [];
  let acc = 0;
  for (const s of segments) { offsets.push(acc); acc += s.end - s.start; }

  const mapTime = (t) => {
    for (let i = 0; i < segments.length; i++) {
      const s = segments[i];
      if (t >= s.start && t <= s.end) return offsets[i] + (t - s.start);
    }
    return null; // word was cut out
  };

  const kept = [];
  for (const w of words) {
    const start = mapTime(w.start);
    const end = mapTime(w.end) ?? (start != null ? start + 0.3 : null);
    if (start != null && end != null) kept.push({ text: w.text, start, end });
  }

  // group into cues of up to ~8 words / ~3.5 s
  const cues = [];
  let cur = null;
  for (const w of kept) {
    if (!cur) cur = { start: w.start, end: w.end, words: [w.text] };
    else if (cur.words.length >= 8 || w.end - cur.start > 3.5) {
      cues.push(cur);
      cur = { start: w.start, end: w.end, words: [w.text] };
    } else {
      cur.words.push(w.text);
      cur.end = w.end;
    }
  }
  if (cur) cues.push(cur);

  return { srt: cuesToSrt(cues), cues };
}

function cuesToSrt(cues) {
  return cues.map((c, i) =>
    `${i + 1}\n${srtTime(c.start)} --> ${srtTime(c.end)}\n${c.words.join(" ")}\n`
  ).join("\n");
}

function srtTime(sec) {
  const ms = Math.floor((sec % 1) * 1000);
  const s = Math.floor(sec) % 60;
  const m = Math.floor(sec / 60) % 60;
  const h = Math.floor(sec / 3600);
  const pad = (n, l = 2) => String(n).padStart(l, "0");
  return `${pad(h)}:${pad(m)}:${pad(s)},${pad(ms, 3)}`;
}

// ---- video cutting with ffmpeg.wasm ------------------------------------
async function loadFFmpeg() {
  if (ffmpeg) return ffmpeg;
  ffmpeg = new FFmpeg();
  ffmpeg.on("progress", ({ progress }) => {
    if (progress >= 0 && progress <= 1) setProgress(0.7 + 0.28 * progress);
  });
  const base = "https://cdn.jsdelivr.net/npm/@ffmpeg/core-mt@0.12.6/dist/esm";
  await ffmpeg.load({
    coreURL: await toBlobURL(`${base}/ffmpeg-core.js`, "text/javascript"),
    wasmURL: await toBlobURL(`${base}/ffmpeg-core.wasm`, "application/wasm"),
    workerURL: await toBlobURL(`${base}/ffmpeg-core.worker.js`, "text/javascript"),
  });
  return ffmpeg;
}

async function cutVideo(file, segments) {
  const ff = await loadFFmpeg();
  const ext = (file.name.split(".").pop() || "mp4").toLowerCase();
  const inName = `in.${ext}`;
  await ff.writeFile(inName, await fetchFile(file));

  // Re-encode each kept segment to its own file, then concat them. This is
  // robust (no fragile filter strings) and cuts accurately at any point.
  const parts = [];
  for (let i = 0; i < segments.length; i++) {
    const { start, end } = segments[i];
    const out = `part${i}.mp4`;
    setDetail(`Trimming clip ${i + 1} of ${segments.length}…`);
    await ff.exec([
      "-i", inName,
      "-ss", start.toFixed(3),
      "-to", end.toFixed(3),
      "-c:v", "libx264", "-preset", "veryfast",
      "-c:a", "aac",
      "-avoid_negative_ts", "make_zero",
      out,
    ]);
    parts.push(out);
  }

  setDetail("Joining the clips…");
  const list = parts.map((p) => `file '${p}'`).join("\n");
  await ff.writeFile("list.txt", new TextEncoder().encode(list));
  await ff.exec(["-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", "out.mp4"]);

  const data = await ff.readFile("out.mp4");
  // clean up the virtual filesystem
  for (const p of [inName, "list.txt", "out.mp4", ...parts]) {
    try { await ff.deleteFile(p); } catch { /* ignore */ }
  }
  return new Blob([data.buffer], { type: "video/mp4" });
}

// ---- step 4: results ---------------------------------------------------
function showResults(videoBlob, srtResult, words, originalDur, segments) {
  hide(els.progress);
  setProgress(1);

  const videoUrl = URL.createObjectURL(videoBlob);
  els.resultVideo.src = videoUrl;
  els.dlVideo.href = videoUrl;
  els.dlVideo.download = "snip-" + (currentFile.name.replace(/\.[^.]+$/, "") || "video") + ".mp4";

  if (srtResult && srtResult.srt) {
    const srtUrl = URL.createObjectURL(new Blob([srtResult.srt], { type: "text/plain" }));
    els.dlSrt.href = srtUrl;
    els.dlSrt.download = "snip-captions.srt";
    show(els.dlSrt);
    renderTranscript(srtResult.cues);
  } else {
    hide(els.dlSrt);
  }

  const keptDur = segments.reduce((a, s) => a + (s.end - s.start), 0);
  const saved = Math.max(0, originalDur - keptDur);
  els.summary.textContent =
    `Original ${fmt(originalDur)} → ${fmt(keptDur)} · trimmed ${fmt(saved)}` +
    (words.length ? ` · ${words.length} words captioned` : "");

  show(els.results);
}

function renderTranscript(cues) {
  els.transcript.innerHTML = "";
  for (const c of cues) {
    const div = document.createElement("span");
    div.className = "cue";
    div.innerHTML = `<span class="t">${fmt(c.start)}</span>${c.words.join(" ")}`;
    els.transcript.appendChild(div);
  }
}

function fmt(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}
