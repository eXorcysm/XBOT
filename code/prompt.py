"""

This module contains functions for building the model prompts.

"""

### Importing libraries ###

import sys

### Environment settings ###

sys.path.append("../")

PROMPT = "./data/characters/xbot/character.txt"

### Module functions ###

def build_character_message(card, bot = None, usr = None):
    header  = "### FIRST ###"
    footer  = "### END ###"
    message = card[card.index(header) + len(header):card.index(footer)].format(character = bot, user = usr)

    return message.strip()

def build_character_persona(card, bot = None, usr = None):
    header  = "### CHARACTER ###"
    footer  = "### USER ###"
    persona = card[card.index(header) + len(header):card.index(footer)].format(character = bot, user = usr)

    return persona.strip()

def build_example_dialogue(card, bot = None, usr = None):
    header  = "### EXAMPLE ###"
    footer  = "### FIRST ###"
    example = card[card.index(header) + len(header):card.index(footer)].format(character = bot, user = usr)

    return example.strip()

def build_model_instructions(card, bot = None, usr = None):
    header = "### INSTRUCT ###"
    footer = "### CHARACTER ###"
    inst   = card[card.index(header) + len(header):card.index(footer)].format(character = bot, user = usr)

    return inst.strip()

def build_scenario(card, bot = None, usr = None):
    header = "### SCENE ###"
    footer = "### EXAMPLE ###"
    scene  = card[card.index(header) + len(header):card.index(footer)].format(character = bot, user = usr)

    return scene.strip()

def build_system_prompt(bot, usr):
    with open(PROMPT, encoding = "utf-8") as txt:
        card = txt.read()

    example   = build_example_dialogue(card, bot, usr)
    first_msg = build_character_message(card, bot, usr)

    prompt  = build_model_instructions(card, bot, usr) + "\n\n"
    prompt += build_character_persona(card, bot, usr)  + "\n\n"
    prompt += build_user_persona(card, bot, usr)       + "\n\n"
    prompt += build_scenario(card, bot, usr)

    return prompt, example, first_msg

def build_user_persona(card, bot = None, usr = None):
    header  = "### USER ###"
    footer  = "### SCENE ###"
    persona = card[card.index(header) + len(header):card.index(footer)].format(character = bot, user = usr)

    return persona.strip()

if __name__ == "__main__":
    prompt, example, first_msg = build_system_prompt(bot = "XBOT", usr = "USER")

    print(prompt)
    print("\n----------\n")
    print(example)
    print("\n----------\n")
    print(first_msg)
