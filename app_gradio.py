import gradio as gr
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from janus.models import MultiModalityCausalLM, VLChatProcessor
from PIL import Image
from threading import Thread
from deepseek_janus import init_deepseek_janus, generate_multimodal_understanding
from prompts import default_image_prompt, default_clinic_prompt
import json
import requests

import numpy as np
import os
import time
# import spaces  # Import spaces for ZeroGPU compatibility


# Load model and processor
janus_vl_chat_processor, janus_vl_gpt, janus_tokenizer = init_deepseek_janus()

def call_final_model(user_prompt, temperature, top_p, max_new_tokens):
    TGI_URL = "http://127.0.0.1:8090/generate_stream"
    MODEL_ID = "aaditya/OpenBioLLM-Llama3-70B"

    messages = [
        {
            "role": "user",
            "content": user_prompt
        },
    ]

    # 🔷 BUILD PROMPT
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    # 🔷 BUILD TGI REQUEST
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False if temperature == 0.0 else True,
            "temperature": temperature,
            "top_p": top_p,
        }
    }

    headers = {
        "Content-Type": "application/json"
    }

    # 🔷 CALL TGI
    with requests.post(TGI_URL, json=payload, headers=headers, stream=True) as resp:
        resp.raise_for_status()  # check for HTTP error

        # iterate over the response line by line
        for line in resp.iter_lines(decode_unicode=True):
            if not line:  # skip empty keep-alives
                continue
            try:
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                data = json.loads(line)
                # print(data)  # handle the JSON object
                if not data['token']['special']:
                    new_token = data['token']['text']
                    yield new_token
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse line: {line!r} error: {e}")



@torch.inference_mode()
def multimodal_understanding(image, question, seed, top_p, temperature):
    return generate_multimodal_understanding(image, question, seed, top_p, temperature, 
                                             janus_vl_chat_processor, janus_vl_gpt, janus_tokenizer)


def get_final_output(image_understanding, prompt_template, temperature, top_p, max_new_tokens):
    final_output = ''
    prompt = prompt_template.replace("<input>", image_understanding)
    for token in call_final_model(prompt, temperature, top_p, max_new_tokens):
        final_output += token
        yield final_output


# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(value="# Multimodal Understanding")
    with gr.Row():
        image_input = gr.Image()
        with gr.Column():
            question_input = gr.Textbox(label="Question", value=default_image_prompt)
            und_seed_input = gr.Number(label="Seed", precision=0, value=42)
            top_p = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.05, label="top_p")
            temperature = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.05, label="temperature")
        with gr.Column():
            final_model_prompt_template = gr.Textbox(label="Probe Prompt", value=default_clinic_prompt)
            final_model_max_new_tokens = gr.Slider(minimum=128, maximum=1024, value=256, step=128, label="med model max new tokens")
            final_model_top_p = gr.Slider(minimum=0, maximum=1, value=0.9, step=0.05, label="med model top_p")
            final_model_temperature = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.05, label="med model temperature")
        
    understanding_button = gr.Button("Chat")
    understanding_output = gr.Textbox(label="Response")
    final_output = gr.Textbox(label="Final Response")

    examples_inpainting = gr.Examples(
        label="Multimodal Understanding examples",
        examples=[
        ],
        inputs=[question_input, image_input],
    )
    
    understanding_button.click(
        multimodal_understanding,
        inputs=[image_input, question_input, und_seed_input, top_p, temperature],
        outputs=understanding_output
    ).then(
        get_final_output,
        inputs=[understanding_output, final_model_prompt_template, final_model_temperature, final_model_top_p, final_model_max_new_tokens],
        outputs=final_output
    )

demo.queue(concurrency_count=1).launch(server_port=8089, share=False)
# demo.queue(concurrency_count=1, max_size=10).launch(server_name="0.0.0.0", server_port=37906, root_path="/path")