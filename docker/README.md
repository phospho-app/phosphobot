# phospho Dockerfiles

## Prerequisites

You need to have Docker installed.

## viewer

This Dockerfile is inspired by https://huggingface.co/spaces/lerobot/visualize_dataset/blob/main/Dockerfile

It is used to run the Hugginggface LeRobot dataset viewer locally (for local or private datasets).

To run it:

```bash
docker build -t viewer -f Dockerfile.viewer .
docker run --name viewer -p 7860:7860 viewer
```

If you want to access private datasets, you need to have a Huggingface account and a token.

You can get the token from https://huggingface.co/settings/tokens

Then, you can run the viewer with the token:

```bash
docker run --name viewer -p 7860:7860 -e HF_TOKEN="your_token" viewer
```
