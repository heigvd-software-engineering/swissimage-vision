#!/bin/bash
# Script inspired by
# https://dev.to/pwd9000/create-a-docker-based-self-hosted-github-runner-linux-container-48dh

set -e  # Exit on error

GH_OWNER=heigvd-software-engineering
GH_REPOSITORY=swissimage-vision

# Set the runner token (exprires after 1 hour)
set_token() {
    REG_TOKEN=$(curl -sX POST -H "Accept: application/vnd.github.v3+json" -H "Authorization: token ${GITHUB_RUNNER_PAT}" https://api.github.com/repos/${GH_OWNER}/${GH_REPOSITORY}/actions/runners/registration-token | jq .token --raw-output)
}

# Configure the runner
set_token
./config.sh --unattended \
    --url https://github.com/${GH_OWNER}/${GH_REPOSITORY} \
    --replace --labels ${GITHUB_RUNNER_LABELS} --token ${REG_TOKEN}

# Cleanup the runner
cleanup() {
    echo "Removing runner..."
    set_token
    ./config.sh remove --unattended --token ${REG_TOKEN}
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

# Start the runner
./run.sh > run.log 2>&1 & wait $!
