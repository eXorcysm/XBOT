# XBOT AI

Named after one of the Three Robots from the *Love, Death & Robots* series, XBOT is a large language model (LLM) chatbot built with retrieval-augmented generation (RAG) enabled. As interaction progresses, XBOT periodically saves chat history and rebuilds her vector database upon restart. As such, her knowledge and awareness of the user will continue to grow with engagement.

**NOTE:** This application is designed for LOCAL deployment only! Future versions will be adapted for cloud hosting.

## Installation

1. Create new virtual environment:

```bash
conda create -n xbot python=3.12
conda activate xbot
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

## Usage

Run application from command line:

```bash
python app.py
```

## References

- [Building and Deploying a Gradio UI on Hugging Face Spaces](https://academy.towardsai.net/courses/take/beginner-to-advanced-llm-dev/multimedia/59791752-building-and-deploying-a-gradio-ui-on-hugging-face-spaces)
- [Enabling Conversational Memory in LLMs](https://academy.towardsai.net/courses/take/beginner-to-advanced-llm-dev/multimedia/59791737-enabling-conversational-memory-in-llms)
- [How to Build a RAG-Powered Interactive Chatbot with Llama3, LlamaIndex, and Gradio](https://www.superteams.ai/blog/steps-to-build-a-rag-powered-interactive-chatbot-with-llama3-llamaindex-and-gradio)
- [How to Create a Chatbot with Gradio](https://www.gradio.app/guides/creating-a-chatbot-fast)
- [Using a Vector Database](https://academy.towardsai.net/courses/take/beginner-to-advanced-llm-dev/multimedia/59791115-using-a-vector-database)
