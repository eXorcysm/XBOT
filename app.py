"""

This is the main module to launch the XBOT app.

"""

### Importing libraries ###

import logging
import sys

from code.bot              import XBOT
from code.prompt           import build_system_prompt
from llama_index.core.llms import MessageRole

import gradio as gr

### Environment settings ###

AI   = "XBOT"
USER = "USER"

logging.basicConfig(
    stream = sys.stdout, level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s"
)

### Module functions ###

def launch_ui():
    """
    Launch chat user interface.
    """

    with gr.Blocks(
        analytics_enabled = True,
        fill_height       = True,
        theme             = "hmb/amethyst",
        title             = "XBOT 🤖"
    ) as chat:
        gr.Markdown(
            "<style>h1 {text-align: center; display: block;}</style><h1>XBOT Chat 🤖</h1>"
        )

        xbot = gr.State(lambda: XBOT(bot_id = AI, user_id = USER))

        _, _, intro = build_system_prompt(bot = AI, usr = USER)

        gradio_bot = gr.Chatbot(
            placeholder      = intro,
            scale            = 1,
            show_copy_button = True,
            show_label       = False,
            type             = "messages"
        )

        gr.ChatInterface(
            additional_inputs = [xbot],
            chatbot           = gradio_bot,
            description       = "[!] Write in third person past tense voice. Enclose *actions* in asterisks and \"speech\" in quotations.",
            fn                = send_query,
            textbox           = gr.Textbox(placeholder = "What's on your mind?"),
            type              = "messages"
        )

        chat.queue(default_concurrency_limit = 32)
        chat.launch(debug = True, share = False)

def send_query(query, history, xbot):
    """
    Send user query to chatbot and return response.
    """

    logging.info("[+] Sending query for %s: %s", xbot.user, query)

    # Print out memory contents for debugging purposes.
    xbot.display_memory()

    chat_memory = xbot.memory.get()  # list of ChatMessage() objects in chat memory buffer

    # Align saved chat history with current session history.
    if len(chat_memory) > 0:
        query_indices = [i for i, q in enumerate(chat_memory) if q.role == MessageRole.USER]

        print("-_" * 50)
        print("Query length:", len(query_indices))
        print("History length:", len(history))
        print("-_" * 50)

        if len(query_indices) > len(history):
            queries_to_delete = query_indices[len(history)]
            chat_memory       = chat_memory[:queries_to_delete]

            print("To be deleted:", queries_to_delete)
            print("-_" * 50)
            print("Bot memory:", chat_memory)
            print("-_" * 50)

            xbot.memory.set(chat_memory)

    logging.info("Chat history: %i --- %s", len(xbot.memory.get()), xbot.memory.get())
    logging.info("Session history: %i --- %s", len(history), history)

    # Send user query to chatbot.
    answer = xbot.chat(query, stream=True)

    # Print out response nodes for debugging purposes.
    xbot.display_nodes(answer)

    # Stream chatbot response.
    tokens = ""

    for token in answer.response_gen:
        tokens += token

        yield tokens

def main():
    """
    Launch application user interface and activate chatbot.
    """

    print("\n========== Welcome to XBOT Chat ==========\n")

    launch_ui()

### TODO
#
# 1. Determine how to add new nodes to vector store without having to rebuild entire store.
# 2. Is there a way to erase prefix messages?
# 3. Insert example and first message into memory (or chat store) so that it can be eventually removed.
# 4. Add delete() method to delete previous message.
# 5. Add reset() method to delete chat history.

if __name__ == "__main__":
    main()
