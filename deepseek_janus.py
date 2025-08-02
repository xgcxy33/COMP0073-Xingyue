from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image

import torch
import numpy as np

def init_deepseek_janus(hf_cache_dir='./hf_home/'):
    deepseek_model_path = "deepseek-ai/Janus-Pro-7B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(deepseek_model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        deepseek_model_path, trust_remote_code=True, cache_dir=hf_cache_dir
    )
    vl_gpt = vl_gpt.to(torch.float32).eval().cpu()
    return vl_chat_processor, vl_gpt, tokenizer


@torch.inference_mode()
def generate_multimodal_understanding(image, question, seed, top_p, temperature, 
                             vl_chat_processor, vl_gpt, tokenizer):
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    # pil_images = [Image.fromarray(image)]
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to('cpu', dtype=torch.float32)
    
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer