# RouteFM project page

This directory contains the zero-build static project page published at
<https://lamda-model-reuse.github.io/RouteFM/>.

Preview the exact Pages artifact locally from the repository root:

```bash
./scripts/build_website.sh /tmp/routefm-site
python3 -m http.server --directory /tmp/routefm-site 8000
```

The live demo is embedded from
<https://huggingface.co/spaces/AIGNLAI/RouteFM-Demo>. The page remains useful
if the Space is sleeping or unavailable because all paper and package links are
ordinary static HTML.
