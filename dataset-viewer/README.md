# LeRobot Dataset Viewer

This dataset viewer is a fork of the [LeRobot Dataset Visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset) to add support for private datasets and other features.

## Setup

```bash
npm install
```

Get a Huggingface token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

Create a `.env.local` file:

```
NEXT_PUBLIC_HUGGINGFACE_TOKEN=your_hf_token_here
```

Run dev server:

```bash
npm run dev
```

## Security

WARNING: This is for local use only. It's not meant to be deployed as it can expose your Huggingface token to the public.
