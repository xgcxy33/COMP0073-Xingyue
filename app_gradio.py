import gradio as gr
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from janus.models import MultiModalityCausalLM, VLChatProcessor
from PIL import Image
from threading import Thread
from deepseek_janus import init_deepseek_janus, generate_multimodal_understanding
from tgi_utils import get_tgi_stream

import prompts
import json
import requests

import numpy as np
import os
import time


# Load model and processor
janus_vl_chat_processor, janus_vl_gpt, janus_tokenizer = init_deepseek_janus()
ULTRASOUND_GUIDE_MODEL_ID = "aaditya/Llama3-OpenBioLLM-70B"


@torch.inference_mode()
def multimodal_understanding(image, question, seed, top_p, temperature):
    return generate_multimodal_understanding(image, question, seed, top_p, temperature, 
                                             janus_vl_chat_processor, janus_vl_gpt, janus_tokenizer)


def get_final_output(image_understanding, prompt_template, temperature, top_p, max_new_tokens):
    final_output = ''
    prompt = prompt_template.replace("<input>", image_understanding)
    for token in get_tgi_stream(user_prompt=prompt, 
                                temperature=temperature, 
                                top_p=top_p, 
                                model_id=ULTRASOUND_GUIDE_MODEL_ID,
                                max_new_tokens=max_new_tokens):
        final_output += token
        yield final_output


def select_image_prompt(bp_area: str):
    image_model_prompt_key = bp_area.lower() + "_image_prompt"
    return getattr(prompts, image_model_prompt_key)


# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(value="# Brachial Plexus Ultrasound Probe Guidance with AI Assistance")
    with gr.Row():
        image_input = gr.Image()
        with gr.Column():
            block_area_dropdown = gr.Dropdown(
                ["Interscalene", "Supraclavicular", "Infraclavicular", "Axillary"],
                label="Choose BP area",
                value="Interscalene"
            )
            with gr.Accordion("Image Understanding Model Advanced Settings", open=False):
                question_input = gr.Textbox(label="Image Understanding Prompt", lines=10)
                und_seed_input = gr.Number(label="Seed", precision=0, value=42)
                top_p = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.05, label="Top P")
                temperature = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.05, label="Temperature")
            demo.load(fn=select_image_prompt, inputs=block_area_dropdown, outputs=question_input)
            block_area_dropdown.change(fn=select_image_prompt, inputs=block_area_dropdown, outputs=question_input)
        with gr.Column():
            with gr.Accordion("Ultrasound Guide Model Advanced Settings", open=False):
                final_model_prompt_template = gr.Textbox(label="Ultrasound Guide Prompt (<input> will be replaced by image understanding model output)", value=prompts.default_clinic_prompt, lines=30)
                final_model_max_new_tokens = gr.Slider(minimum=128, maximum=2048, value=1024, step=128, label="Max New Tokens")
                final_model_top_p = gr.Slider(minimum=0, maximum=1, value=0.9, step=0.05, label="Top P")
                final_model_temperature = gr.Slider(minimum=0, maximum=1, value=0.9, step=0.05, label="Temperature")
        
    understanding_button = gr.Button("Generate Ultrasound Probe Guidance")
    understanding_output = gr.Textbox(label="Intermediate Response - Image Understanding")
    final_output = gr.Textbox(label="Final Response - Ultrasound Probe Guidance")

    examples_inpainting = gr.Examples(
        label="Examples",
        examples=[
            [
                "./images/Interscalene/00.png",
                "Interscalene",
            ],
            [
                "./images/Supraclavicular/10.png",
                "Supraclavicular"
            ],
            [
                "./images/Infraclavicular/20.png",
                "Infraclavicular"
            ],
            [
                "./images/Axillary/30.png",
                "Axillary"
            ]
        ],
        inputs=[image_input, block_area_dropdown],
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

demo.queue(concurrency_count=1).launch(server_port=8089, share=True)