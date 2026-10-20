
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model
model_name = "sshleifer/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def summarize(text):
    if not text or not text.strip():
        return "Please enter some text to summarize."

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


# Create the Gradio interface
demo = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(
        lines=10,
        label="Input Text",
        placeholder="Paste an article or paragraph here..."
    ),
    outputs=gr.Textbox(
        lines=5,
        label="Summary"
    ),
    title="AI Text Summarizer",
    description="Summarize long text using the DistilBART model.",
    examples=[
        [
            "Lions are social animals that live in groups called prides. "
            "Lionesses cooperate to hunt prey, while males help defend "
            "the territory. Wild lions live mainly in sub-Saharan Africa "
            "and in Gir National Park in India. They face threats from "
            "habitat loss and human-wildlife conflict."
        ]
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)
