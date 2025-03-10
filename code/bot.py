"""

This module instantiates the chatbot with the XBOT class.

"""

### Importing libraries ###

import datetime
import logging
import os
import sys

from llama_index.core.llms import MessageRole
from .engine               import init_chat_engine
from .engine               import init_chat_settings
from .memory               import build_local_index
from .memory               import init_chat_memory
from .memory               import init_chat_store
from .memory               import load_local_index
from .memory               import parse_text_nodes

### Environment settings ###

sys.path.append("../")

CHAT_HISTORY = "./data/chat_history/"
CHAT_STORE   = "./data/chat_store.json"
VECTOR_STORE = "./data/vector_store/"

### Class definition ###

class XBOT():
    """
    This class instantiates the chatbot.
    """

    def __init__(self, bot_id, user_id):
        self.ai         = bot_id
        self.chat_store = init_chat_store(CHAT_STORE)
        self.memory     = init_chat_memory(self.chat_store, user_id)
        self.user       = user_id

        # Initialize XBOT settings.
        init_chat_settings()

        # Build/rebuild vector store index if chat history exists.
        chat_dir = os.listdir(CHAT_HISTORY)

        if len(chat_dir) > 0:
            nodes = parse_text_nodes(CHAT_HISTORY)

            if nodes:
                build_local_index(nodes, VECTOR_STORE)
            else:
                sys.exit("ERROR: cannot build vector store index without data nodes!")

        # Initialize vector index (from vector store, if available).
        if os.path.isfile(VECTOR_STORE + "docstore.json"):
            self.index = load_local_index(VECTOR_STORE)
        else:
            self.index = None

        # Initialize chat engine.
        self.chat_engine = init_chat_engine(bot_id, user_id, self.memory, self.index)

    def chat(self, query, limit_pct = 0.8, stream = False):
        """
        Send user query to chat engine and collect response.
        """

        # Flush chat memory buffer when token count exceeds threshold.
        chat_memory = self.memory.get()
        token_count = self.memory._token_count_for_messages(chat_memory)

        if token_count > limit_pct * self.memory.token_limit:
            self.flush_memory()

        if stream:
            answer = self.chat_engine.stream_chat(query)
        else:
            answer = self.chat_engine.chat(query)

        logging.info("[+] Got response from model: %s", answer)

        # Save message exchange to chat store.
        self.chat_store.persist(persist_path = CHAT_STORE)

        return answer

    def display_memory(self):
        """
        Print contents of chat memory buffer.
        """

        chat_memory = self.memory.get()

        print("\n", "-_" * 50, "\n")
        print("===== CURRENT TOKEN COUNT =====\n")

        token_count = self.memory._token_count_for_messages(chat_memory)

        print(token_count, "/", self.memory.token_limit)
        print("\n", "-_" * 50, "\n")
        print("===== CHAT MEMORY =====\n")

        messages = ""

        for _, msg in enumerate(chat_memory):
            if msg.role == MessageRole.USER and msg.content is not None:
                messages += self.user + ": " + msg.content.strip() + "\n"
            elif msg.role == MessageRole.ASSISTANT and msg.content is not None:
                messages += self.ai + ": " + msg.content.strip() + "\n\n"

        print(messages)
        print("\n", "-_" * 50, "\n")

    def display_nodes(self, answer):
        """
        Print response nodes retrieved from vector store.
        """

        print("\n", "-_" * 50, "\n")
        print("===== RETRIEVAL RESULTS =====\n\t", answer.response.replace("\n", ""))
        print("Sources nodes:\n")

        if answer.source_nodes:
            for src in answer.source_nodes:
                print("\tNode ID\t", src.node_id)
                print("\tText\t", src.text)
                print("\tScore\t", src.score)
                print("\n\t" + "-_" * 50, "\n")
        else:
            print("\tNo sources from vector store used!\n")

    def flush_memory(self, limit_pct = 0.5):
        """
        Free up chat memory buffer.
        """

        chat_memory  = self.memory.get()
        flush_memory = []
        token_count  = self.memory._token_count_for_messages(chat_memory)
        token_limit  = self.memory.token_limit

        while token_count > limit_pct * token_limit:
            # Flush oldest chats from memory.
            flush_memory.append(chat_memory.pop(0))

            token_count = self.memory._token_count_for_messages(chat_memory)

        self.memory.set(chat_memory)

        # Write flushed memory to disk.
        self.record(flush_memory)

        logging.info("[+] New token count: %i / %i", token_count, token_limit)

    def record(self, chat_memory, output_path = CHAT_HISTORY):
        """
        Write chat messages in current memory to disk.
        """

        messages = ""

        for _, msg in enumerate(chat_memory):
            if msg.role == MessageRole.USER and msg.content is not None:
                messages += self.user + ": " + msg.content.strip() + "\n"
            elif msg.role == MessageRole.ASSISTANT and msg.content is not None:
                messages += self.ai + ": " + msg.content.strip() + "\n"

        output_file  = output_path + "history_"
        output_file += datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt"

        with open(output_file, encoding = "utf-8", mode = "w") as buf:
            buf.write(messages)

        return len(messages)
