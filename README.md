# Project Setup Guide

Follow the steps below to install and run this project locally.  

---

## Prerequisites

Before you begin, make sure you have:

- **Git** installed  
- **Python 3.8+** installed  
- **pip** (Python package manager) installed  
- **Docker** installed

---

## Installation Steps

### 1. Clone the repository
```bash
git clone git@github.com:xgcxy33/COMP0073-Xingyue.git
cd COMP0073-Xingyue
```

### 2. Install Git (if not already installed)
On linux
```bash
sudo apt-get update && sudo apt-get install -y git
```

### 3. Install Python (>= 3.8)
Check Python version:
```bash
python3 --version
```
If you need to install:

On Linux
```bash
sudo apt-get install -y python3 python3-pip
```

### 4. Install Docker

Follow the office Docker guide to install Docker CLI https://docs.docker.com/engine/install/

### 5. Install dependencies

```bash
pip3 install -r requirements.txt
```

## Setup Hugging Face TGI (Text Generation Inference)

This project requires Hugging Face’s Text Generation Inference (TGI) server to host a [`Llama3-OpenBioLLM-70B`](https://huggingface.co/aaditya/Llama3-OpenBioLLM-70B) model on port 8090.

This LLM usually requires multiple GPUs or HPUs. Example command to run on Intel Gaudi Accelerator
```bash
volume=$PWD/data
model=aaditya/OpenBioLLM-Llama3-70B

docker run -d -v $volume:/data  -p 8090:80 --runtime=habana  -e HABANA_VISIBLE_DEVICES=4,5 -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
   -e TEXT_GENERATION_SERVER_IGNORE_EOS_TOKEN=false \
   -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
   -e MAX_TOTAL_TOKENS=4096 \
   -e BATCH_BUCKET_SIZE=256 \
   -e PREFILL_BATCH_BUCKET_SIZE=4 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=64 \
   -e ENABLE_HPU_GRAPH=true \
   -e LIMIT_HPU_GRAPH=true \
   -e USE_FLASH_ATTENTION=true \
   -e FLASH_ATTENTION_RECOMPUTE=true \
   --cap-add=sys_nice \
   --ipc=host \
   ghcr.io/huggingface/text-generation-inference:latest-gaudi \
   --model-id $model \
   --sharded true --num-shard 2 \
   --max-batch-size 2 \
   --max-input-length 2048 --max-total-tokens 4096 \
   --max-batch-prefill-tokens 4096 --max-batch-total-tokens 524288 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 5
```

If you have limited GPUs/HPUs, but have enough RAM, it is recommended to use a smaller model [`Llama3-Med42-8B`](https://huggingface.co/m42-health/Llama3-Med42-8B)
```bash
volume=$PWD/data
model=m42-health/Llama3-Med42-8B

docker run -d \
  --rm \
  -v $volume:/data \
  -p 8090:80 \
  --ipc=host \
  --shm-size 1g \
  ghcr.io/huggingface/text-generation-inference:3.3.4-intel-cpu \
  --model-id $model \
  --max-input-tokens 4096 \
  --max-total-tokens 8192 \
  --cuda-graphs 0
```

Ensure the TGI runs successfully, you can check the docker log
```bash
docker logs -f <docker_container_id>
```
Once you see this log line
```aiignore
INFO text_generation_router::server: router/src/server.rs:2298: Connected
```
You can ping TGI by using `curl`
```bash
curl http://localhost:8090/health
```

## Run the Gradio App
Start the application:
```bash
python3 app_gradio.py
```

## Access the Application
Open your browser at: http://localhost:8090

Or, copy the generated public Gradio URL from output and open it in your browser.




