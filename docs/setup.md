# Setup

- [Overview](#overview)
- [Repository Configuration](#repository-configuration)
  - [Access](#access)
  - [Branch protection rules](#branch-protection-rules)
  - [Repository Secrets](#repository-secrets)
  - [Self-hosted GitHub Runner](#self-hosted-github-runner)

## Overview

This page explains the configuration and setup that was used in the repository. The repository uses GitHub Actions to run the CI/CD pipeline and a self-hosted GitHub runner to execute the workflows on a on-premises Kubernetes cluster.

## Repository Configuration

### Access

Make sure to configure the repository access settings to restrict access to the repository to only the necessary users.

### Branch protection rules

In your repository, go to `Settings` -> `Branches`. Add `main` branch protection rules to the repository settings. The rules should include:

- Check "Require a pull request before merging"
- Uncheck "Allow force pushes"
- Uncheck "Allow deletions"

This ensures that all changes are reviewed and pass the CI/CD pipeline before being merged into the main branch.

This also prevents from configuring the CI/CD pipeline to run on the main branch, which can cause the pipeline twice when merging a pull request.

### Repository Secrets

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

### Self-hosted GitHub Runner

We use a self-hosted GitHub runner to execute the GitHub Action workflows on a on-premises Kubernetes cluster. The runner listens for jobs from GitHub Actions and creates Kubernetes jobs to execute the workflows using CML.

To see more information about configuring a self-hosted GitHub runner, see the [../infra/github-runner/README.md](../infra/github-runner/README.md) file.