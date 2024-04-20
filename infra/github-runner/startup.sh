#!/bin/bash

# Configure the runner
./config.sh --unattended \
    --url https://github.com/heigvd-software-engineering/swissimage-vision \
    --replace --labels $GITHUB_RUNNER_LABELS --token $GITHUB_RUNNER_TOKEN

# Start the runner
./run.sh
