# RouteFM project page

This directory contains the zero-build static project page published at
<https://lamda-model-reuse.github.io/RouteFM/>.

Preview the exact Pages artifact locally from the repository root:

```bash
./scripts/build_website.sh /tmp/routefm-site
python3 -m http.server --directory /tmp/routefm-site 8000
```

The free static result explorer is embedded from
<https://huggingface.co/spaces/AIGNLAI/RouteFM-Demo>. The page remains useful
if the Space is sleeping or unavailable because all paper and package links are
ordinary static HTML.

## Paper videos

The `#video` section contains a single English/Chinese player. Use `?video=zh#video`
for a direct link to the Chinese version. Playback does not start automatically;
switching languages pauses the current clip before changing its source.

The build downloads reviewed MP4, SRT, and WebVTT assets from the dedicated
`paper-videos-v1` GitHub release using `scripts/fetch_paper_videos.py`.
`docs/paper_videos.json` pins each asset's SHA-256; mismatches stop the build.
The files are deployed with Pages for same-origin playback and captions, while
large MP4s stay out of Git history. English and Chinese posters are small
tracked images in `assets/`. Previewing the exact artifact therefore needs
network access the first time, or an existing verified media cache.
