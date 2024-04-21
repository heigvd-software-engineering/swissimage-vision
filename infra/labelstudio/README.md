# Label Studio

- [Overview](#overview)
- [Install Label Studio](#install-label-studio)
  - [Helm Configuration](#helm-configuration)
  - [Installation on Kubernetes](#installation-on-kubernetes)
  - [Project Configuration](#project-configuration)
- [Uninstall Label Studio](#uninstall-label-studio)

## Overview

This page provides instructions on how to install and configure Label Studio on a Kubernetes cluster. Label Studio is a data labeling tool that allows you to create labeled datasets for machine learning models.

## Install Label Studio

In this section, we will install Label Studio on a Kubernetes cluster using Helm. If you wish to install Label Studio locally, you can use `minikube` or follow the instructions in the [official documentation](https://labelstud.io/guide/install.html).

### Helm Configuration

In this directory, create a `ls-values.yaml` file with the following configuration. Make sure to replace the placeholders with your own values.

```yaml
app:
  ingress:
    enabled: true
    host: <YOUR_HOST>
  extraEnvironmentVars:
    {
      "LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK": "true",
      "LABEL_STUDIO_USERNAME": "<YOUR_USERNAME_EMAIL>",
      "LABEL_STUDIO_PASSWORD": "<YOUR_SECRET_PASSWORD>",
      "SSRF_PROTECTION_ENABLED": "true",
      "collect_analytics": "false",
    }
```

### Installation on Kubernetes

Add the Helm chart repository to install and update Label Studio.

```bash
helm repo add heartex https://charts.heartex.com/
helm repo update heartex
```

Install Label Studio using Helm.

```bash
helm install label-studio heartex/label-studio -f ls-values.yaml
```

### Project Configuration

To configure the Label Studio project, two options are available:

1. Manually create a project in the Label Studio UI.
2. Use the Label Studio API.

In this directory, you can find a Python script that uses the Label Studio API to configure a project. To use the script,

1. Navigate to the root of the repository.
2. Include the following environment variables in a `.env` file.

   ```bash
   AWS_S3_ENDPOINT=...
   AWS_ACCESS_KEY_ID=...
   AWS_SECRET_ACCESS_KEY=...

   LABEL_STUDIO_HOST=...
   LABEL_STUDIO_TOKEN=...
   ```

> [!NOTE]
> The `LABEL_STUDIO_TOKEN` can be found in the Label Studio UI under `Account & Settings` in the top right corner.

3. Install the required dependencies.

   ```bash
   pip install -r infra/labelstudio/requirements.txt
   ```

4. Run the script.

   ```bash
   pip install -r infra/labelstudio/requirements.txt
   python3 infra/labelstudio/configure.py
   ```

Now, you can access the Label Studio UI at the host you specified in the `ls-values.yaml` file.

## Uninstall Label Studio

Uninstall Label Studio using Helm.

```bash
helm delete label-studio
```
