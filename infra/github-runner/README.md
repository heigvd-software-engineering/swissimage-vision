# Self-hosted GitHub Runner

- [Overview](#overview)
- [Configuring the Repository](#configuring-the-repository)
- [Deploy GitHub Runner](#deploy-github-runner)
- [Action Workflow Configuration](#action-workflow-configuration)
  - [Node affinity](#node-affinity)

## Overview

This guide explains how to deploy a self-hosted GitHub runner to a Kubernetes cluster. The runner is used to execute the GitHub Action workflows defined in the repository.

## Configuring the Repository

> [!CAUTION]
> Creating a self-hosted runner allows other users to execute code on your infrastructure. Make sure to secure your runner and restrict access to the repository.

1. Disable running workflows from fork pull requests.

   In the repository, go to `Settings -> Actions` and disable `Fork pull request workflows`.

2. Create self host runner on the GitHub repository. See [GitHub documentation](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners) for more information.

## Deploy GitHub Runner

1. Deploy runner to Kubernetes cluster:

   ```bash
   kubectl apply -f infra/github-runner/runner.yaml
   ```

2. Configure runner with the GitHub repository:

   ```bash
   # Connect to the runner pod
   kubectl exec -it github-runner -- bash
   ```

   ```bash
   # Run the configuration script
   cd actions-runner
   ./config.sh --token <your-token>
   ```

   > [!NOTE]
   > You might be prompted to install some extra dependencies. Follow the instructions and re-run the configure script.

3. Start the runner with the following command:

   ```bash
   nohup ./run.sh &> runner.log &
   ```

   This will start the runner in the background. You can check the logs with the following command:

   ```bash
   tail -f runner.log
   ```

## Action Workflow Configuration

### Node affinity

In the `.github/workflows/mlops.yaml` file, we have defined a node affinity to run the CML job on a node with GPU. You might need to adjust the node affinity based on your cluster configuration.

```bash
cml runner \
   --labels="cml-runner" \
   --cloud="kubernetes" \
   --cloud-kubernetes-node-selector="<your-selector>" \
   --single
```
