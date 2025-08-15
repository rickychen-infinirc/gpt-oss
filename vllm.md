uv venv --python 3.12 --seed
source .venv/bin/activate
sudo snap install astral-uv --classic

conda install -c conda-forge libstdcxx-ng
conda install -c conda-forge gxx_linux-64
conda install gcc_linux-64=12.3.0 gxx_linux-64=12.3.0 -c conda-forge




uv pip install --pre vllm==0.10.1+gptoss \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  --index-strategy unsafe-best-match

export C_INCLUDE_PATH="/usr/include:/usr/local/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="/usr/include:/usr/local/include:$CPLUS_INCLUDE_PATH"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/lib:$LIBRARY_PATH"

export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
vllm serve openai/gpt-oss-20b --async-scheduling




export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
vllm serve /home/rickychen/Desktop/llm/models/gpt-oss-20b \
  --tensor-parallel-size 2 \
  --port 30000 \
  --host 0.0.0.0 \
  --async-scheduling \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 24 \
  --max-model-len 20480 \
  --served-model-name gpt-oss


-----
# 添加NVIDIA Container Toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 更新並安裝
sudo apt update
sudo apt install -y nvidia-container-toolkit

# 配置Docker使用NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker



# 2. 拉取官方映像
docker pull vllm/vllm-openai:gptoss

# 3. 啟動服務
docker run --gpus all \
  -p 30000:8000 \
  --ipc=host \
  -v /home/rickychen/Desktop/llm/models/gpt-oss-20b:/models/gpt-oss-20b \
  -e VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  vllm/vllm-openai:gptoss \
  --model /models/gpt-oss-20b \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port 8000 \
  --async-scheduling
