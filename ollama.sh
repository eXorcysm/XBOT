#!/bin/bash

# Start Ollama server as background process.
ollama serve &

# Wait for server to warm up.
while ! nc -z localhost 7860; do
    echo "[+] Waiting for Ollama server to start ..."

    sleep 1
done

# Pull LLM from Hugging Face and run.
echo "[+] Running LLM ..."

ollama run hf.co/backyardai/Fimbulvetr-11B-v2-GGUF:Q6_K

echo "[+] Ollama server ready!"

# Keep container running.
wait
