"""

This module is responsible for initializing the chat engine and settings.

"""

### Importing libraries ###

import logging
import os
import sys

from llama_index.core                                  import Settings
from llama_index.core.chat_engine                      import SimpleChatEngine
from llama_index.core.llms                             import ChatMessage
from llama_index.core.llms                             import MessageRole
from llama_index.embeddings.huggingface                import HuggingFaceEmbedding
from llama_index.llms.ollama                           import Ollama
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from .prompt                                           import build_system_prompt

### Environment settings ###

sys.path.append("../")

CHAT_HISTORY   = "./data/chat_history/"
CHUNK_OVERLAP  = 24
CHUNK_SIZE     = 512
EMBED_MODEL    = "hoangthethief/best_model"
MAX_NEW_TOKENS = 256
MODEL_NAME     = "hf.co/backyardai/Fimbulvetr-11B-v2-GGUF:Q6_K"
RERANK_MODEL   = "BAAI/bge-reranker-base"
VECTOR_STORE   = "./data/vector_store/"

logging.basicConfig(
    stream = sys.stdout, level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s"
)

### Module functions ###

def init_chat_engine(ai, user, chat_memory = None, index = None):
    """
    Initialize XBOT chat engine.
    """

    instruct, example, first_msg = build_system_prompt(bot = ai, usr = user)

    if example:
        template = instruct + "\n\n" + example + "\n\n" + first_msg
    else:
        template = instruct + "\n\n" + first_msg

    messages = [
        ChatMessage(role = MessageRole.SYSTEM, content = instruct),
        ChatMessage(role = MessageRole.ASSISTANT, content = first_msg)
    ]

    if index is None:
        chat_engine = SimpleChatEngine.from_defaults(
            chat_mode       = "condense_plus_context",
            llm             = Settings.llm,
            memory          = chat_memory,
            prefix_messages = messages,
            verbose         = True
        )
    else:
        reranker = FlagEmbeddingReranker(model = RERANK_MODEL, top_n = 3)

        chat_engine = index.as_chat_engine(
            chat_mode           = "condense_plus_context",
            llm                 = Settings.llm,
            memory              = chat_memory,
            node_postprocessors = [reranker],
            similarity_top_k    = 3,
            system_prompt       = template,
            verbose             = True
        )

    return chat_engine

def init_chat_settings():
    """
    Initialize XBOT environment and model settings.
    """

    logging.info("[+] Initializing XBOT settings ...")

    # Initialize global model settings.
    Settings.chunk_overlap = CHUNK_OVERLAP
    Settings.chunk_size    = CHUNK_SIZE
    Settings.embed_model   = HuggingFaceEmbedding(model_name = EMBED_MODEL)
    Settings.llm           = Ollama(model = MODEL_NAME, request_timeout = 60.0)
    Settings.num_output    = MAX_NEW_TOKENS

    # Initialize local environment directories.
    if not os.path.exists(CHAT_HISTORY):
        os.makedirs(CHAT_HISTORY)

    if not os.path.exists(VECTOR_STORE):
        os.makedirs(VECTOR_STORE)

    logging.info("[+] XBOT initialization complete!")
