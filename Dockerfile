FROM ollama/ollama:latest

# Create user and grant permissions.
RUN useradd -ms /bin/bash ollama

# Set home directory.
ENV HOME=/home/ollama

# Set working directory.
WORKDIR $HOME

# Ensure user has access to home directory.
RUN mkdir -p $HOME/.ollama && chown -R ollama:ollama $HOME/.ollama

# Install netcat (nc) for server readiness.
RUN apt-get update && apt-get install -y netcat

# Copy ollama script before switching users.
COPY ollama.sh /usr/local/bin/ollama.sh

# Set permissions for ollama script.
RUN chmod +x /usr/local/bin/ollama.sh

# Switch to non-root user.
USER ollama

# Set Ollama to listen on all network interfaces.
ENV OLLAMA_HOST=0.0.0.0:7860

# Expose Ollama port.
EXPOSE 7860

# Set Ollama script as entry point.
ENTRYPOINT ["/usr/local/bin/ollama.sh"]
