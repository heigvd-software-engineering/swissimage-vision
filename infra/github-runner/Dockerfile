FROM ubuntu:22.04
LABEL org.opencontainers.image.source="https://github.com/heigvd-software-engineering/swissimage-vision"

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential lsb-release \
    python3 python3-pip \
    curl jq vim gpg wget \
    git unzip tar gettext-base

# Add a non-root user
RUN useradd -m runner

WORKDIR /home/actions-runner

# Install GitHub Actions Runner
RUN curl -o actions-runner-linux-x64-2.315.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.315.0/actions-runner-linux-x64-2.315.0.tar.gz
RUN tar xzf ./actions-runner-linux-x64-2.315.0.tar.gz
# Install extra dependencies for the runner
RUN ./bin/installdependencies.sh

COPY startup.sh .

RUN chmod +x startup.sh

USER runner

ENTRYPOINT ["./startup.sh"]
