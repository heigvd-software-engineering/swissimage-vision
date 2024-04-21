# Self-hosted GitHub Runner

- [Overview](#overview)
- [Configuring the Repository](#configuring-the-repository)
- [Deploy GitHub Runner](#deploy-github-runner)
- [Building the Docker image](#building-the-docker-image)
- [GPU Runner](#gpu-runner)
  - [Alternatives](#alternatives)
- [Resources](#resources)

## Overview

This page explains how to deploy a self-hosted GitHub runner to a Kubernetes cluster. The runner is used to execute the GitHub Action workflows defined in the repository.

The runner uses a custom Docker image that includes the necessary dependencies to run the workflows. The Docker image is built and pushed to the GitHub Container Registry.

## Configuring the Repository

> [!CAUTION]
> Creating a self-hosted runner allows other users to execute code on your infrastructure. Make sure to secure your runner and restrict access to the repository.


**Disable running workflows from fork pull requests**

In the repository, go to `Settings -> Actions` and disable `Fork pull request workflows`.


## Deploy GitHub Runner

First, you need to create a Kubernetes secret to store a personal access token (PAT) in order to create the runner.

The PAT is required to have the following permissions:

- `repo` - to access the repository

Run the following command to create the secret:

```bash
printf "Enter your GitHub runner PAT: " && read TOKEN \
   && kubectl create secret generic github-runner-pat --from-literal=token=$TOKEN
```

To deploy runner to Kubernetes cluster, run the following command:

```bash
kubectl apply -f runner.yaml
```

This will deploy a GitHub runner pod named `github-runner` in your current namespace.

You can check the runner logs by connecting to the pod:

```bash
kubectl exec -it github-runner -- bash
tail -f run.log
```

## Building the Docker image

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

## GPU Runner

We also have a similar yaml file for GPU runner (`runner-gpu.yaml`). This is used within the workflow `train-and-report.yaml` to create a self-hosted GPU runner only for executing the needed steps. This has the advantage of only utilizing the GPU resources when needed.

### Alternatives

CML also provides a way to deploy a self-hosted runner using the `cml runner` command. However, this method has downsides:
- Docker-in-Docker is required to increase the shared memory size, which can be a security risk.
- The runner does not have resources limits, it is limited to node affinity.

## Resources

https://dev.to/pwd9000/create-a-docker-based-self-hosted-github-runner-linux-container-48dh
