# Video generation API setup

The promotion-video pipeline will keep provider credentials outside source
control. Do not paste an API key into chat, a GitHub issue, or a tracked file.

## Configure credentials

From the RouteFM repository root:

```bash
cp .env.video.example .env.video
chmod 600 .env.video
```

Edit `.env.video` locally and set:

```dotenv
VIDEO_API_KEY=your_real_key
VIDEO_API_BASE=https://provider.example/v1
VIDEO_MODEL=provider-model-name
```

The real `.env.video` file is ignored by Git. To check configuration without
printing the secret:

```bash
set -a
source .env.video
set +a
python3 -c 'import os; print("configured" if os.getenv("VIDEO_API_KEY") else "missing")'
```

## Information still needed

Send only the non-secret integration details:

1. official API documentation URL;
2. text-to-video and/or image-to-video endpoint;
3. model name and supported duration, resolution, and aspect ratio;
4. asynchronous job creation and polling schema;
5. rate limits and expected credit cost;
6. whether generated outputs expire and must be downloaded immediately.

Once these are known, the provider-specific client, storyboard prompts,
polling, download, and final `ffmpeg` assembly can be implemented without
changing how the secret is stored.
