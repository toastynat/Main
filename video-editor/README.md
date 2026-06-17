# ✂️ Snip — auto-cut silences & caption your videos

Drop in a video, and Snip trims out the silent gaps and writes captions for
you. **Everything runs inside your web browser** — on your phone or your
laptop — so your video never gets uploaded anywhere, and there's nothing to
install and no cost.

## How to use it

### The easy way (once it's online)

When this folder is published to GitHub Pages, you'll have a link like:

```
https://toastynat.github.io/Main/video-editor/
```

Open that link on your laptop or your phone, choose a video, and tap
**"Make my video."** When it's done you can download the trimmed clip and a
captions file.

> **First run is slower:** the first time you make captions, it downloads a
> small AI model (the "tiny" one is ~40 MB). After that it's cached and quick.

### Trying it on your own laptop first

You can also run it straight from your computer:

1. Open the Terminal app.
2. Go into this folder and start a tiny local web server:
   ```
   cd Main/video-editor
   python3 -m http.server 8000
   ```
3. Open **http://localhost:8000** in Chrome.

(It needs to be served by a web server like this — opening `index.html`
directly as a file won't work, because browsers block the video engine in that
mode.)

## What you get

- **Trimmed video** — the silent gaps cut out, as an `.mp4` you can download.
- **Captions** — a `.srt` subtitle file. Most video apps, YouTube, and phone
  players can load an `.srt` to show captions. (Burning the words directly
  onto the picture, TikTok-style, is the planned next step — see below.)

## Options

- **How quiet counts as silence** — drag toward −20 dB to cut only near-total
  silence, or toward −60 dB to be more aggressive.
- **Shortest gap to cut** — ignore pauses shorter than this so speech doesn't
  get chopped.
- **Caption model** — `tiny` is fastest; `base` is more accurate but a bigger
  download. There's also an "any language" option.

## Good to know

- Works best with **clips up to a few minutes**. Very long videos can run out
  of memory in a phone browser — use a laptop for those, or cut them down
  first.
- It needs a reasonably modern browser (recent Chrome, Edge, Safari, or
  Firefox).

## How it works under the hood

1. **Read the audio** with the browser's built-in Web Audio API.
2. **Find the quiet bits** by measuring loudness in small windows and marking
   stretches below your threshold.
3. **Transcribe** the audio with [Whisper](https://github.com/openai/whisper),
   running locally via
   [transformers.js](https://github.com/huggingface/transformers.js).
4. **Re-cut the video** with [ffmpeg.wasm](https://ffmpegwasm.netlify.app/) and
   hand you the result.

## Ideas for later

- Burn captions directly onto the video (karaoke / TikTok style).
- A fast "cloud mode" for long videos.
- Filler-word removal ("um", "uh").
- Adjustable caption style and position.
