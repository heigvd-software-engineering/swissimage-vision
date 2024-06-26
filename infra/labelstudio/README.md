# Label Studio

- [Overview](#overview)
- [Install Label Studio](#install-label-studio)
  - [Helm Configuration](#helm-configuration)
  - [Installation on Kubernetes](#installation-on-kubernetes)
  - [Project Configuration](#project-configuration)
- [Uninstall Label Studio](#uninstall-label-studio)
- [Migrating Annotations](#migrating-annotations)

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

> [!NOTE]
> Your can find an exhaustive list of configuration options in the [official chart repository](https://github.com/HumanSignal/charts/tree/master/heartex/label-studio).

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

In this directory, you can find a Python script (`configure.py`) that uses the Label Studio API to configure a project. To use the script,

1. Update the `configure.py` script to match your needs. You can find the documentation for the Label Studio API SDK [here](https://labelstud.io/sdk/index.html) with an exhaustive list of available methods and configuration options.
2. Update the `label_config.xml` file with your own labeling configuration. You can find more information about the Label Studio labeling configuration [here](https://labelstud.io/tags).
3. Navigate to the root of the repository.
4. Include the following environment variables in a `.env` file.

   ```bash
   AWS_S3_ENDPOINT=...
   AWS_ACCESS_KEY_ID=...
   AWS_SECRET_ACCESS_KEY=...

   LABEL_STUDIO_HOST=...
   LABEL_STUDIO_TOKEN=...
   ```

> [!NOTE]
> The `LABEL_STUDIO_TOKEN` can be found in the Label Studio UI under `Account & Settings` in the top right corner.

4. Install the required dependencies.

   ```bash
   pip install -r infra/labelstudio/requirements.txt
   ```

5. Run the script.

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

## Migrating Annotations

One limitation of saving the annotations to S3, is that they contain Label Studio project, task and annotations IDs. This means that if you want to migrate the annotations to a different Label Studio instance, you will need to update these IDs and create the annotations in the Label Studio database. You can find the `migration_utils` directory with a Python script that can help you with this process.
