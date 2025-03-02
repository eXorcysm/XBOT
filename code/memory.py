"""

This module is responsible for managing chatbot memory.

"""

### Importing libraries ###

import argparse
import logging
import os
import sys
import chromadb

from llama_index.core                    import Settings
from llama_index.core                    import SimpleDirectoryReader
from llama_index.core                    import StorageContext
from llama_index.core                    import VectorStoreIndex
# from llama_index.core                    import load_index_from_storage
# from llama_index.core.memory             import VectorMemory
from llama_index.core.memory             import ChatSummaryMemoryBuffer
# from llama_index.core.node_parser        import SimpleNodeParser
# from llama_index.core.readers.json       import JSONReader
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.embeddings.huggingface  import HuggingFaceEmbedding
from llama_index.vector_stores.chroma    import ChromaVectorStore
from transformers                        import AutoTokenizer

### Environment settings ###

sys.path.append("../")

CHROMA      = "xbot_chat"
DATA_DST    = "./data/characters/xbot/vector_store/"
DATA_SRC    = "./data/characters/xbot/chat_history/"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOKEN_MODEL = "Sao10K/Fimbulvetr-11B-v2"

Settings.embed_model = HuggingFaceEmbedding(EMBED_MODEL)
Settings.tokenizer   = AutoTokenizer.from_pretrained(TOKEN_MODEL)

logging.basicConfig(
    stream = sys.stdout, level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s"
)

### Module functions ###

def build_local_index(nodes, store_path = DATA_DST):
    """
    Build index over nodes and save locally in vector store.
    """

    chroma_db      = chromadb.PersistentClient(path = store_path + CHROMA)
    chroma_collect = chroma_db.get_or_create_collection(name = CHROMA)
    store          = ChromaVectorStore(chroma_collection = chroma_collect)
    context        = StorageContext.from_defaults(vector_store = store)

    index = VectorStoreIndex.from_documents(
        nodes,
        embed_model     = Settings.embed_model,
        show_progress   = True,
        storage_context = context
    )

    index.storage_context.persist(persist_dir = store_path)

def init_chat_memory(store, user_id):
    """
    Initialize chat memory buffer.
    """

    chat_memory = ChatSummaryMemoryBuffer.from_defaults(
        chat_store     = store,
        chat_store_key = user_id.lower(),
        token_limit    = 1024  # 3072
    )

    return chat_memory

def init_chat_store(chat_store):
    """
    Initialize chat store for memory buffer.
    """

    if os.path.isfile(chat_store):
        return SimpleChatStore.from_persist_path(persist_path = chat_store)

    return SimpleChatStore()

# def load_local_index(vector_store):
#     """
#     Load index from local storage.
#     """

#     store = StorageContext.from_defaults(persist_dir = vector_store)

#     if "default" not in store.vector_stores:
#         store.vector_stores["default"] = vector_store

#     index = load_index_from_storage(show_progress = True, storage_context = store)

#     return index

def load_local_index(vector_store):
    """
    Load index from local storage.
    """

    chroma_db      = chromadb.PersistentClient(path = vector_store + CHROMA)
    chroma_collect = chroma_db.get_or_create_collection(name = CHROMA)
    store          = ChromaVectorStore(chroma_collection = chroma_collect)

    index = VectorStoreIndex.from_vector_store(
        embed_model   = Settings.embed_model,
        show_progress = True,
        use_async     = True,
        vector_store  = store
    )

    return index

def parse_cmd_args():
    """
    Configure command line argument parser.
    """

    parser = argparse.ArgumentParser(usage = "python " + sys.argv[0])

    parser.add_argument(
        "-d", "--dst", type = str, default = DATA_DST, help = "document destination path"
    )

    # parser.add_argument(
    #     "-j", "--json", action = "store_true", help = "build/rebuild vector index from JSON files"
    # )

    parser.add_argument(
        "-s", "--src", type = str, default = DATA_SRC, help = "document source path"
    )

    # parser.add_argument(
    #     "-t", "--text", action = "store_true", help = "build/rebuild vector index from text files"
    # )

    parser.add_argument(
        "-v", "--verbose", action = "store_true", help = "show details of build process"
    )

    return parser.parse_args()

# def parse_json_nodes(src = DATA_SRC, verbose = False):
#     """
#     Load chat history from store and parse into nodes.
#     """

#     reader    = JSONReader()
#     documents = reader.load_data(input_file = src + "store.json")

#     return documents

#     # Parse documents into nodes.
#     parser = SimpleNodeParser.from_defaults(chunk_overlap = 16, chunk_size = 128)
#     nodes  = parser.get_nodes_from_documents(documents)

#     # Manually set node identifiers.
#     for i, node in enumerate(nodes):
#         node.id_ = "node_" + str(i)

#     if verbose:
#         print("Total number of nodes:", len(nodes))

#     return nodes

def parse_text_nodes(src = DATA_SRC, verbose = False):
    """
    Load chat history and parse into nodes.
    """

    reader    = SimpleDirectoryReader(input_dir = src)
    documents = reader.load_data()

    # Parse documents into nodes (not compatible with Chroma).
    # parser = SimpleNodeParser.from_defaults(chunk_overlap = 16, chunk_size = 128)
    # nodes  = parser.get_nodes_from_documents(documents)

    nodes = documents

    # Manually set node identifiers.
    # for i, node in enumerate(nodes):
    #     node.id_ = "node_" + str(i)

    if verbose:
        logging.info("[+] Total number of nodes parsed: %i", len(nodes))

    return nodes

def main():
    """
    Module's main function is called to index XBOT's memory.
    """

    print("\n========== XBOT Data Processing Module ==========\n")

    # Configure command line argument parser.
    args = parse_cmd_args()

    # Initialize local environment directories.
    if not os.path.exists(DATA_DST):
        os.makedirs(DATA_DST)

    if not os.path.exists(DATA_SRC):
        os.makedirs(DATA_SRC)

    # Prepare data source.
    nodes = None

    # if args.json:
    #     nodes = parse_json_nodes(args.src, args.verbose)
    # elif args.text:
    #     nodes = parse_text_nodes(args.src, args.verbose)

    nodes = parse_text_nodes(args.src, args.verbose)

    if nodes:
        build_local_index(nodes, args.dst)
    else:
        sys.exit("ERROR: cannot build vector store index without data nodes!")

    logging.info("[+] Local index build complete!")

if __name__ == "__main__":
    main()
