"""

This is the main module to launch the XBOT app.

"""

### Importing libraries ###

import logging
import sys

from code.bot    import XBOT
from code.prompt import build_system_prompt

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
        theme             = "gstaff/sketch",
        title             = "XBOT 🤖"
    ) as chat:
        gr.Markdown(
            "<style>h1 {text-align: center; display: block;}</style><h1>XBOT Chat 🤖</h1>"
        )

        xbot = gr.State(lambda: XBOT(bot_id = AI, user_id = USER))

        _, _, intro = build_system_prompt(bot = AI, usr = USER)

        instruct  = "[!] Write in third person past tense voice."
        instruct += r" Enclose \*actions\* in asterisks"
        instruct += " and \"speech\" in quotation marks."

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
            description       = instruct,
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
    # xbot.display_memory()

    logging.info("Chat history: %i --- %s", len(xbot.memory.get()), xbot.memory.get())
    logging.info("Session history: %i --- %s", len(history), history)

    # Send user query to chatbot.
    answer = xbot.chat(query, stream = True)

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

if __name__ == "__main__":
    main()
