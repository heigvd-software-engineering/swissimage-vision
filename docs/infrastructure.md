# Infrastructure

- [Overview](#overview)
- [Development Workflow](#development-workflow)
- [Action Workflows](#action-workflows)
- [Reference](#reference)
  - [Configuration](#configuration)
    - [Access](#access)
    - [Branch protection rules](#branch-protection-rules)
    - [Repository Secrets](#repository-secrets)
    - [Self-hosted GitHub Runner](#self-hosted-github-runner)

## Overview

This guide explains the infrastructure used in the repository and how to configure it.

## Development Workflow

<img src="../media/infra.png" width="650" />

1. The developer pushes code into an active pull request.
2. The GitHub Action workflow is triggered.
3. The self-hosted GitHub runner creates another self-hosted runner with GPU and runs the `train-and-report` step.
4. `train-and-report`:
   1. Runs the DVC pipeline and reports the results back to the pull request
   2. Pushes the results to the DVC remote storage
   3. Pushes the updated DVC lock file to the repository
5. The developer reviews the results and merges the pull request.

## Action Workflows

The GitHub Action workflows are defined in the `.github/workflows` directory.

The workflow `train-and-report.yaml` is triggered when a pull request is opened or updated. The workflow runs the following steps:

- `setup-runner` - Create a self-hosted gpu runner
- `train-and-report` - Runs the DVC pipeline and reports the results back to the pull request
- `cleanup-runner` - Deletes the self-hosted gpu runner

## Reference

### Configuration

#### Access

Make sure to configure the repository access settings to restrict access to the repository to only the necessary users.

#### Branch protection rules

In your repository, go to `Settings` -> `Branches`. Add `main` branch protection rules to the repository settings. The rules should include:

- Check "Require a pull request before merging"
- Uncheck "Allow force pushes"
- Uncheck "Allow deletions"

This ensures that all changes are reviewed and pass the CI/CD pipeline before being merged into the main branch.

This also prevents from configuring the CI/CD pipeline to run on the main branch, which can cause the pipeline twice when merging a pull request.

#### Repository Secrets

In order to run the workflow, you need to configure the repository secrets. The following secrets are required:

`AWS_ACCESS_KEY_ID`

The AWS access key ID used by DVC.

`AWS_S3_ENDPOINT`

The AWS S3 endpoint used by DVC.

`AWS_SECRET_ACCESS_KEY`

The AWS secret access key used by DVC.

`KUBECONFIG`

The Kubernetes configuration file to access the Kubernetes cluster.

> [!NOTE]
> Make sure to set the Kubernetes namespace to yours in the context of the `KUBECONFIG` file.

#### Self-hosted GitHub Runner

We use a self-hosted GitHub runner to execute the GitHub Action workflows on a on-premises Kubernetes cluster. The runner listens for jobs from GitHub Actions and creates Kubernetes jobs to execute the workflows using CML.

To see more information about configuring a self-hosted GitHub runner, see the [../infra/github-runner/README.md](../infra/github-runner/README.md) file.
