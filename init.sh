
echo "Initializing environment..."

echo "Cloning and building oneDNN..."
git clone https://github.com/CUTE-Project/oneDNN.git
cd oneDNN
git checkout v3.9
mkdir build
cd build
cmake ..
cmake --build . -j 32
cd ../..
echo "Build completed successfully." 


echo "Setting up OpenVINO environment..."
python -m venv openvino_env_2025.2
source ./openvino_env_2025.2/bin/activate
pip install --upgrade pip
pip install openvino==2025.2.0
echo "OpenVINO environment setup completed."

echo "Setting up LLM Benchmark environment..."
python -m venv ov-llm-bench-env
source ./ov-llm-bench-env/bin/activate
git clone https://github.com/CUTE-Project/openvino.genai.git
cd openvino.genai/tools/llm_bench
pip install -r requirements.txt  
echo "LLM Benchmark environment setup completed."


echo "Downloading models..."

echo "Downloading LLama3.2-1B Model"
echo "huggingface need to be logged in to download models."
huggingface-cli login
optimum-cli export openvino --model meta-llama/Llama-3.2-1B-Instruct --weight-format int8 ./model/llama.model/openvino/Llama-3.2-1B-Instruct-Q8
optimum-cli export openvino --model meta-llama/Llama-3.2-1B-Instruct --weight-format fp16 ./model/llama.model/openvino/Llama-3.2-1B-Instruct-Q16
optimum-cli export openvino --model meta-llama/Llama-3.2-1B-Instruct --quant-mode    int8 ./model/llama.model/openvino/Llama-3.2-1B-Instruct-Q8-all
echo "LLama3.2-1B Model download completed."

echo "Downloading ResNet-50 Bert Model"
git clone https://github.com/CUTE-Project/model_zoo_Quantitative.git
echo "if you want quantize model by yourself, please refer to https://github.com/openvinotoolkit/open_model_zoo or https://github.com/CUTE-Project/open_model_zoo"
echo "ResNet-50 Bert Model download completed."

echo "All models downloaded successfully."

echo "Environment initialization completed. Ready to run tests~ :)"
echo "Environment initialization completed. Ready to run tests~ :)"
echo "Environment initialization completed. Ready to run tests~ :)"