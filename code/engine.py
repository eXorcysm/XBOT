"""

This module is responsible for initializing the chat engine and settings.

"""

### Importing libraries ###

import logging
import os
import sys

# from llama_index.core                         import PromptTemplate
# from llama_index.core                         import ServiceContext
from llama_index.core                         import Settings
# from llama_index.core                         import SimpleDirectoryReader
# from llama_index.core                         import StorageContext
# from llama_index.core                         import VectorStoreIndex
# from llama_index.core                         import load_index_from_storage
from llama_index.core.chat_engine             import SimpleChatEngine
from llama_index.core.llms                    import ChatMessage
from llama_index.core.llms                    import MessageRole

# from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface       import HuggingFaceEmbedding
from llama_index.llms.ollama                  import Ollama
from .prompt                                  import build_system_prompt

### Environment settings ###

sys.path.append("../")

CHAT_HISTORY   = "./data/characters/xbot/chat_history/"
CHUNK_OVERLAP  = 24
CHUNK_SIZE     = 128  # 512
MAX_NEW_TOKENS = 384
MODEL_NAME     = "hf.co/backyardai/Fimbulvetr-11B-v2-GGUF:Q6_K"
VECTOR_STORE   = "./data/characters/xbot/vector_store/"

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

    # template = PromptTemplate(instruct + "\n\n" + first_msg)
    # vector_memory = build_vector_memory(index.vector_store)

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
            # system_prompt   = template,
            verbose         = True
        )
    else:
        chat_engine = index.as_chat_engine(
            chat_mode        = "condense_plus_context",
            llm              = Settings.llm,
            memory           = chat_memory,
            similarity_top_k = 3,
            system_prompt    = template,
            verbose          = True
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
    Settings.embed_model   = HuggingFaceEmbedding(model_name = "BAAI/bge-small-en-v1.5")
    Settings.llm           = Ollama(model = MODEL_NAME, request_timeout = 60.0)
    Settings.num_output    = MAX_NEW_TOKENS

    # Initialize local environment directories.
    if not os.path.exists(CHAT_HISTORY):
        os.makedirs(CHAT_HISTORY)

    if not os.path.exists(VECTOR_STORE):
        os.makedirs(VECTOR_STORE)

    logging.info("[+] XBOT initialization complete!")
