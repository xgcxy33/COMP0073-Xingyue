from deepseek_janus import init_deepseek_janus, generate_multimodal_understanding
from tgi_utils import get_tgi_stream
from datetime import datetime

import argparse
import json
import prompts
import os


def generate_combined_model_response(image, args, vl_chat_processor, vl_gpt, tokenizer):
    print(f"Generating image caption for {image}")
    image_caption_prompt = getattr(prompts, args.image_model_prompt_key)
    print(f"Using image caption prompt: {image_caption_prompt}")
    image_caption = generate_multimodal_understanding(image=image, 
                                                      question=image_caption_prompt, 
                                                      seed=args.image_model_seed, 
                                                      top_p=args.image_model_top_p, 
                                                      temperature=args.image_model_temperature, 
                                                      vl_chat_processor=vl_chat_processor, 
                                                      vl_gpt=vl_gpt, 
                                                      tokenizer=tokenizer)
    print(f"Generated image_caption: {image_caption}")
    clinic_model_prompt_template = getattr(prompts, args.clinic_model_prompt_key)
    clinic_model_prompt = clinic_model_prompt_template.replace("<input>", image_caption)
    print(f"Using clinic model prompt: {clinic_model_prompt}")
    clinic_model_output_generator = get_tgi_stream(user_prompt=clinic_model_prompt, 
                   temperature=args.clinic_model_temperature, 
                   top_p=args.clinic_model_top_p, 
                   model_id=args.clinic_model_id, 
                   tgi_port=args.clinic_model_tgi_port)
    clinic_model_output = ""
    for token in clinic_model_output_generator:
        clinic_model_output += token
    print(f"Clinic model output: {clinic_model_output}")
    return clinic_model_output
    
    
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("input_file_path", help="Input file path")
    parser.add_argument("--image_model_top_p", default=0.95)
    parser.add_argument("--image_model_temperature", default=0.5)
    parser.add_argument("--image_model_seed", default=42)
    parser.add_argument("--image_model_prompt_key", default="default_image_prompt")
    parser.add_argument("--clinic_model_top_p", default=0.9)
    parser.add_argument("--clinic_model_temperature", default=0.9)
    parser.add_argument("--clinic_model_prompt_key", default="default_clinic_prompt")
    parser.add_argument("--clinic_model_id", default="aaditya/Llama3-OpenBioLLM-70B")
    parser.add_argument("--clinic_model_tgi_port", default=8090)
    parser.add_argument("--output_file_path", default=None)

    # Parse the arguments
    args = parser.parse_args()

    vl_chat_processor, vl_gpt, tokenizer = init_deepseek_janus()

    input_file_data = None
    with open(args.input_file_path, 'r') as input_file:
        input_file_data = json.load(input_file)

    res = vars(args)
    res["outputs"] = []
    for record in input_file_data:
        generated_text = generate_combined_model_response(image=record["image_path"], 
                                               args=args,
                                               vl_chat_processor=vl_chat_processor, 
                                               vl_gpt=vl_gpt, 
                                               tokenizer=tokenizer)
        record["clinic_output"] = generated_text
        res["outputs"].append(record)

    output_file_path = args.output_file_path
    if not output_file_path:
        input_filename = os.path.basename(args.input_file_path)
        input_filename = input_filename.split('.', maxsplit=1)[0]
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_file_path = f"./output/{input_filename}_{timestamp}.json"

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, "w") as output_file:
        json.dump(res, output_file, ensure_ascii=False, indent=4)
        
        
        
    