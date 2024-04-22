# Self-hosted GitHub Runner

- [Overview](#overview)
  - [GPU Runner](#gpu-runner)
    - [Alternatives](#alternatives)
- [Install the GitHub Runner](#install-the-github-runner)
  - [Configuration](#configuration)
    - [Repository](#repository)
    - [Runner](#runner)
  - [Building the Docker Image](#building-the-docker-image)
  - [Deploy GitHub Runner](#deploy-github-runner)
- [Uninstalling the GitHub Runner](#uninstalling-the-github-runner)
- [Resources](#resources)

## Overview

This page explains how to deploy a self-hosted GitHub runner to a Kubernetes cluster (`runner.yaml`). The runner is used to execute the GitHub Action workflows defined in the repository.

The runner uses a custom Docker image that includes the necessary dependencies to run the workflows. The Docker image is built and pushed to the GitHub Container Registry. (see [Building the Docker image](#building-the-docker-image))

### GPU Runner

We also have a similar yaml file for GPU runner (`runner-gpu.yaml`). This is used within the workflow `train-and-report.yaml` to create a self-hosted GPU runner only for executing the needed steps. This has the advantage of only utilizing the GPU resources when needed. It also uses the same Docker image as the CPU runner.

#### Alternatives

CML also provides a way to deploy a self-hosted runner using the `cml runner` command. However, this method has downsides:

- Docker-in-Docker is required to increase the shared memory size (needed for model training), which can be a security risk.
- The container runs in privileged mode, which can be a security risk.
- The runner does not have resources limits, it is limited to node affinity.

## Install the GitHub Runner

### Configuration

> [!CAUTION]
> Creating a self-hosted runner allows other users to execute code on your infrastructure. Make sure to secure your runner and restrict access to the repository.

#### Repository

**Disable running workflows from fork pull requests**

In the repository, go to `Settings -> Actions` and disable `Fork pull request workflows`.

#### Runner

You will also need to modify the following for your own repository:

- `GH_OWNER` and `GH_REPOSITORY` in `startup.sh`
- `LABEL` in `Dockerfile`
- `spec.containers.image` in `runner.yaml` and `runner-gpu.yaml`

### Building the Docker Image

> [!NOTE]
> For the next steps, make sure to update the tag of the Docker image to match your repository.

1. Authenticate to the GitHub Container Registry. See [GitHub documentation](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry) for more information.

2. Build docker image

   ```bash
   docker build -t ghcr.io/heigvd-software-engineering/swissimage-vision/github-runner:latest .
   ```

3. Push the docker image to the GitHub Container Registry

   ```bash
   docker push ghcr.io/heigvd-software-engineering/swissimage-vision/github-runner:latest
   ```

> [!NOTE]
> Make sure to set the image visibility to `Public` in the GitHub Container Registry settings.

### Deploy GitHub Runner

First, you need to create a Kubernetes secret to store a personal access token (PAT) in order to create the runner.

The PAT is required to have the following permissions:

- `repo` - to access the repository

Run the following command to create the secret:

```bash
printf "Enter your GitHub runner PAT: " && read TOKEN \
   && kubectl create secret generic github-runner-pat --from-literal=token=$TOKEN
```

To deploy runner to Kubernetes cluster, run navigate to this folder and the following command:

```bash
kubectl apply -f runner.yaml
```

This will deploy a GitHub runner pod named `github-runner` in your current namespace. The runner will automatically register itself to the repository. See `startup.sh` for more information.

You can check the runner logs by connecting to the pod:

```bash
kubectl exec -it github-runner -- bash
tail -f run.log
```

## Uninstalling the GitHub Runner

To remove the runner from the Kubernetes cluster, run the following command:

```bash
kubectl delete -f runner.yaml
```

The runner will automatically be removed from the repository. See `startup.sh` for more information.

## Resources

**GitHub About self-hosted runners** https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners

**GitHub REST API endpoints for self-hosted runners** https://docs.github.com/en/rest/actions/self-hosted-runners?apiVersion=2022-11-28

**Self-hosted GitHub runner with Docker** https://dev.to/pwd9000/create-a-docker-based-self-hosted-github-runner-linux-container-48dh
