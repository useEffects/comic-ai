import base64
import io
import os
from PIL import Image, ImageDraw, ImageFont
import requests
import torch
from transformers import pipeline
import spacy
import textwrap

nlp = spacy.load("en_core_web_sm")

story = """
Title: "The Quest for the Mythical Devil Fruit"

Once upon a time, in the vast expanse of the Grand Line, where adventure and danger were woven together, a notorious pirate crew named the "Mystic Marauders" set sail. Led by the fearless Captain Luffy D. Monkey, the crew was on a quest like no other.

Rumors had reached their ears about a legendary Devil Fruit, known as the "Mythical Gum-Gum Fruit," said to grant unimaginable powers to its eater. Legends whispered that this fruit could bestow Luffy with even more extraordinary abilities, making him the Pirate King.

As the Mystic Marauders roamed from island to island, they encountered formidable foes and made lifelong allies. Zoro, the crew's swordsman, faced off against a swordsman of an opposing pirate crew in a clash that sent shockwaves through the ocean itself. Nami, the navigator, deciphered ancient maps leading them to hidden treasures. Sanji, the cook, proved his worth not only in the kitchen but in battles as well. Usopp's tales of the crew's adventures inspired others to join their cause. Chopper's medical expertise became crucial in times of peril, while Robin's archaeology knowledge unveiled the secrets of the world.

One fateful day, the Mystic Marauders reached the enigmatic "Isle of the Eternal Storm." This cursed island was shrouded in perpetual tempests, and it was rumored to be the resting place of the Mythical Gum-Gum Fruit.

With unwavering determination, the crew set foot on the perilous island. They navigated through treacherous jungles, fought off monstrous creatures, and deciphered cryptic riddles left behind by an ancient civilization. Along the way, they discovered the island's guardian, a colossal sea serpent named Leviathor.

A grand battle ensued, with Luffy using his extraordinary abilities to face Leviathor head-on. With the combined might of his crew, they managed to defeat the guardian and unlock the hidden chamber containing the fruit.

With the Mythical Gum-Gum Fruit in hand, Luffy faced a choice that would change the course of his destiny. Should he consume it and gain immeasurable power, or should he preserve the balance of the world and let it remain untouched?

In the end, Luffy made the decision that defined him as a true Pirate King. He left the fruit untouched, realizing that it was the journey, the camaraderie, and the adventures with his crew that mattered most.

And so, the Mystic Marauders continued their voyage across the Grand Line, forever seeking the greatest treasure of all—the freedom to sail the seas with the ones they cherished most, bound by the bonds of friendship and the thrill of adventure.
"""

summarizer = pipeline(
    "summarization",
    "pszemraj/long-t5-tglobal-base-16384-book-summary",
    device=0 if torch.cuda.is_available() else -1,
)

summarizedText = summarizer(story, max_length=len(story) / 1.5)[0]["summary_text"]
doc = nlp(summarizedText)
for index, sentence in enumerate(doc.sents):
    response = requests.post(
        url=f"http://127.0.0.1:7860/sdapi/v1/txt2img",
        json={
            "prompt": sentence.text,
        },
    )
    data = response.json()
    for i in data["images"]:
        dataImage = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
        image = Image.new("RGB", (dataImage.width, dataImage.height + 80), "white")
        image.paste(dataImage, (0, 0))
        draw = ImageDraw.Draw(image)
        wrapped_text = textwrap.fill(sentence.text, width=(dataImage.width / 8)-20)
        draw.text(
            xy=(10, dataImage.height + 10),
            text=wrapped_text,
            fill=(0, 0, 0),
            font=ImageFont.truetype(font=r"/home/aprilia/.local/share/fonts/CaskaydiaCoveNerdFontMono-Regular.ttf", size=16)
        )
        image_filename = os.path.join("dumps", f"{index}.png")
        image.save(image_filename)
