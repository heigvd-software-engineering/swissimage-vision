# LabelStudio

## Install Label Studio

### Configuration

Create a `ls-values.yaml` file with the following configuration. Make sure to replace the placeholders with your own values.

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

## Uninstall Label Studio

Uninstall Label Studio using Helm.

```bash
helm delete label-studio
```
