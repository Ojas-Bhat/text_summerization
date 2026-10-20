
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "sshleifer/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def summarize(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            min_new_tokens=25,
            num_beams=4,
            do_sample=False
        )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

text = """The lion (Panthera leo) is a powerful big cat in the family
Felidae, famously known as the king of the jungle. Lions have a
muscular build, sharp claws, strong teeth, and a golden-yellow coat.
Adult males have a thick, dark mane. Lions are social animals that
live in prides consisting of females, cubs, and resident males.
Lionesses cooperate to hunt zebras and antelopes, while males help
defend the territory. Wild lions now live mostly in sub-Saharan
Africa and in Gir National Park in Gujarat, India. They are listed
as Vulnerable due to habitat loss and human-wildlife conflict."""

print(summarize(text))
