# Video generation with SinapisAI

RouteFM's promotion-video tooling keeps the SinapisAI credential outside
source control. Never paste an API key into a tracked file, GitHub issue, log,
or command that would store it in shell history.

## Configure credentials

From the RouteFM repository root:

```bash
cp .env.video.example .env.video
chmod 600 .env.video
```

Edit only the local `.env.video` file:

```dotenv
SINAPISAI_API_KEY=your_real_key
SINAPISAI_API_BASE=https://api.sinapisai.com/v1
SINAPISAI_VIDEO_MODEL=doubao/doubao-seedance-2-0-260128
SINAPISAI_POLL_INTERVAL=10
SINAPISAI_TIMEOUT=1800
```

The real `.env.video` file is ignored by Git. Load it into the current shell
and verify its presence without printing the secret:

```bash
set -a
source .env.video
set +a
python3 -c 'import os; print("configured" if os.getenv("SINAPISAI_API_KEY") not in {None, "", "replace_me"} else "missing")'
```

Environment variables loaded this way apply only to the current terminal.
Open a new terminal and you will need to source the file again.

## Submit a video task

Install the small HTTP dependency once:

```bash
python3 -m pip install requests
```

Submitting a request can consume paid API credits. Review the prompt first,
then run:

```bash
python3 scripts/sinapisai_video.py \
  --prompt "Create a concise cinematic research teaser for RouteFM, a foundation model for LLM routing."
```

The client sends `POST /v1/videos`, polls `GET /v1/videos/{id}`, and writes the
latest provider response to `outputs/sinapisai/<task-id>.json`. It never prints
or stores the API key. Use `--submit-only` to create a task without polling.

The literal Python URL is `https://api.sinapisai.com/v1/videos`; the backslash
in `https\://` is only an artifact of escaped rich text and must not appear in
Python source.

## Planned RouteFM video workflow

The production workflow combines the strongest parts of
[paper2video](https://github.com/edwardyen724-g/paper2video) and
[manim-skill](https://github.com/vumichien/manim-skill) rather than relying on
one long generative-video prompt:

1. follow paper2video's paper-to-outline and scene-level storyboard approach;
2. use manim-skill for precise architecture diagrams, equations, labels, and charts;
3. use Seedance through SinapisAI for short cinematic transitions or visual metaphors;
4. assemble narration, captions, Manim scenes, and generated clips with ffmpeg;
5. retain prompts, task JSON, scene sources, and citations for reproducibility.

Before automatic clip download is added, confirm the provider's completed-task
response schema or send an official API documentation link. In particular, we
still need the output URL field, supported duration/resolution/aspect-ratio
parameters, rate limits, credit cost, and output-expiration policy.
