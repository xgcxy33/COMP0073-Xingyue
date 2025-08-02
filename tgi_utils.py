from transformers import AutoTokenizer
import requests
import json


def get_tgi_stream(user_prompt, temperature, top_p, model_id, 
                   max_new_tokens=256, tgi_port=8090):
    tgi_url = f"http://127.0.0.1:{tgi_port}/generate_stream"


    messages = [
        {
            "role": "user",
            "content": user_prompt
        },
    ]

    # 🔷 BUILD PROMPT
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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
    with requests.post(tgi_url, json=payload, headers=headers, stream=True) as resp:
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
                    full_response += new_token
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse line: {line!r} error: {e}")