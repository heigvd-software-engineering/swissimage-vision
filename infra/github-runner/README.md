# Self-hosted GitHub Runner

- [Configuring the Repository](#configuring-the-repository)
- [Deploy GitHub Runner](#deploy-github-runner)


## Configuring the Repository

TODO: Not documented yet.

## Deploy GitHub Runner

1. Create self host runner on the GitHub repository.

2. Deploy runner to Kubernetes cluster:

   ```bash
   kubectl apply -f infra/github-runner/runner.yaml
   ```

3. Configure runner with the GitHub repository:

   ```bash
   kubectl exec -it  github-runner -- bash
   ```

   Inside the container, run the following commands:

   ```bash
   cd actions-runner
   ./config.sh --token <your-token>
   ```

   > You might be prompted to install some extra dependencies. Follow the instructions and re-run the configure script.

   ```bash
   nohup ./run.sh &
   ```
