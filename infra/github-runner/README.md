# Self-hosted GitHub Runner

- [Overview](#overview)
- [Configuring the Repository](#configuring-the-repository)
- [Deploy GitHub Runner](#deploy-github-runner)
- [Action Workflow Configuration](#action-workflow-configuration)
  - [Node affinity](#node-affinity)
- [Building the Docker image](#building-the-docker-image)

## Overview

This page explains how to deploy a self-hosted GitHub runner to a Kubernetes cluster. The runner is used to execute the GitHub Action workflows defined in the repository.

## Configuring the Repository

> [!CAUTION]
> Creating a self-hosted runner allows other users to execute code on your infrastructure. Make sure to secure your runner and restrict access to the repository.

1. Disable running workflows from fork pull requests.

   In the repository, go to `Settings -> Actions` and disable `Fork pull request workflows`.

2. Create self host runner on the GitHub repository. See [GitHub documentation](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners) for more information.

## Deploy GitHub Runner

First, you need to create a Kubernetes secret to store the runner token. Run the following command to create the secret:

```bash
printf "Enter your GitHub runner token: " && read TOKEN \
   && kubectl create secret generic github-runner-token --from-literal=token=$TOKEN
```

To deploy runner to Kubernetes cluster, run the following command:

```bash
kubectl apply -f runner.yaml
```

This will deploy a GitHub runner pod named `github-runner` in your current namespace.

You can check the runner logs with the following command:

```bash
kubectl logs -f github-runner
```

## Action Workflow Configuration

### Node affinity

In the `.github/workflows/train-and-report.yaml` file, we have defined a node affinity to run the CML job on a node with GPU. You might need to adjust the node affinity based on your cluster configuration.

```bash
cml runner \
   --labels="cml-runner" \
   --cloud="kubernetes" \
   --cloud-kubernetes-node-selector="<your-selector>" \
   --single
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
