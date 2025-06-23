from flask import Flask, render_template, request, jsonify, Response
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM

import base64
import requests
from io import BytesIO

from PIL import Image

app = Flask(__name__)

def download_image_url(url):

    # Fetch the image from the URL
    response = requests.get(url)

    # Open the image with Pillow
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        print(f"Failed to download image. Status code: {response.status_code}")


def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """
    buffered = BytesIO()
    pil_image.save(buffered, format=pil_image.format)  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    print(img_str)
    return img_str


def granite_response(user_input):

    model = OllamaLLM(model="granite3.2-vision")
    image = download_image_url(user_input)
    image_b64 = convert_to_base64(image)
    llm_with_image_context = model.bind(images=[image_b64])
    granite_output = llm_with_image_context.invoke("Describe the image")
    print(granite_output)
    return granite_output

# A simple chatbot function (you can modify it to be more complex)
def chatbot_response(user_input: str):
    doctor_input, image_url = user_input.split('|', maxsplit=1)
    granite_output = granite_response(image_url)
    prompt = PromptTemplate(
        input_variables=["doctor_input", "evidence_text"],
        template=(
            "You are a helpful medical assistant. Your task is to suggest follow-up questions for a clinician to ask the patient based on both the doctor's query and medical evidence. "
            "The medical evidence comes from text derived from medical imaging (like an X-ray) or medical test results (like a blood test report). "
            "The follow-up questions should be relevant, thoughtful, and aimed at gathering more information to help diagnose or understand the patient's condition better.\n\n"
            
            "Here is an example:\n\n"
            
            "Evidence (from medical imaging/text): 'The X-ray shows signs of mild pneumonia with some patchy infiltrates in the lower lobe.'\n"
            "Doctor's Query: 'What further symptoms should I ask the patient to identify the severity of this pneumonia?'\n"
            "Clinician Follow-Up Questions:\n"
            "1. 'How long has the patient been experiencing symptoms like cough, fever, or difficulty breathing?'\n"
            "2. 'Has the patient experienced any chest pain or a decrease in oxygen levels?'\n"
            "3. 'Does the patient have a history of respiratory conditions, such as asthma or chronic bronchitis?'\n"
            "4. 'Has the patient had a recent exposure to respiratory infections or traveled to areas with outbreaks?'\n"
            "5. 'Has the patient experienced any weight loss, night sweats, or fatigue?'\n\n"
            
            "Now, based on the following medical evidence and doctor's query, suggest follow-up questions for the clinician to ask:\n\n"
            
            "Evidence (from medical imaging/text): {evidence_text}\n"
            "Doctor's Query: {doctor_input}\n"
        )
    )

    model = OllamaLLM(model="thewindmom/llama3-med42-8b")

    chain = prompt | model

    for t in chain.stream({"doctor_input": user_input, "evidence_text": granite_output}):
        yield t


@app.route('/test')
def test():
    return 'hello'

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/ask', methods=['POST'])
# def ask():
#   user_input = request.json.get('message')
#   return Response(chatbot_response(user_input), content_type='text/plain;charset=utf-8')

@app.route('/ask', methods=['POST'])
def ask_granite():
    user_input = request.form.get('message')
    image_file = request.files.get('image')
    if image_file:
        # 直接用上传的图片
        image = Image.open(image_file.stream)
        image_b64 = convert_to_base64(image)
        granite_output = granite_response_with_b64(image_b64)
    else:
        # 兼容原有逻辑：输入为图片链接
        granite_output = granite_response(user_input.split('|', 1)[1])
    # 下面的逻辑保持不变
    return Response(chatbot_response(user_input), content_type='text/plain;charset=utf-8')

# 新增一个函数，直接用base64图片
def granite_response_with_b64(image_b64):
    model = OllamaLLM(model="granite3.2-vision")
    llm_with_image_context = model.bind(images=[image_b64])
    granite_output = llm_with_image_context.invoke("Describe the image")
    print(granite_output)
    return granite_output

if __name__ == '__main__':
    app.run(debug=True,port=8088)