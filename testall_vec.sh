
echo "Testing Matmul MNK=256 --> 8192"

ONEDNN_MAX_CPU_ISA=AVX10_1_512 taskset -c 0 ../oneDNN/build/tests/benchdnn/benchdnn --mode=p --matmul --dt=u8:s8:s32 --wtag=any 256x256:256x256 256x512:512x256 256x768:768x256 256x1024:1024x256 256x1280:1280x256 256x1536:1536x256 256x1792:1792x256 256x2048:2048x256 256x2304:2304x256 256x2560:2560x256 256x2816:2816x256 256x3072:3072x256 256x3328:3328x256 256x3584:3584x256 256x3840:3840x256 256x4096:4096x256 256x4352:4352x256 256x4608:4608x256 256x4864:4864x256 256x5120:5120x256 256x5376:5376x256 256x5632:5632x256 256x5888:5888x256 256x6144:6144x256 256x6400:6400x256 256x6656:6656x256 256x6912:6912x256 256x7168:7168x256 256x7424:7424x256 256x7680:7680x256 256x7936:7936x256 256x8192:8192x256 > ./vec/matmul/thread1/vec_matmul.csv

# ONEDNN_MAX_CPU_ISA=AVX10_1_512 taskset -c 0-3 ../oneDNN/build/tests/benchdnn/benchdnn --cold-cache=all --mode=p --matmul --dt=u8:s8:s32 --wtag=any 256x256:256x256 256x512:512x256 256x768:768x256 256x1024:1024x256 256x1280:1280x256 256x1536:1536x256 256x1792:1792x256 256x2048:2048x256 256x2304:2304x256 256x2560:2560x256 256x2816:2816x256 256x3072:3072x256 256x3328:3328x256 256x3584:3584x256 256x3840:3840x256 256x4096:4096x256 256x4352:4352x256 256x4608:4608x256 256x4864:4864x256 256x5120:5120x256 256x5376:5376x256 256x5632:5632x256 256x5888:5888x256 256x6144:6144x256 256x6400:6400x256 256x6656:6656x256 256x6912:6912x256 256x7168:7168x256 256x7424:7424x256 256x7680:7680x256 256x7936:7936x256 256x8192:8192x256 > ./vec/matmul/thread4/vec_matmul.csv

source ../openvino_env_2025.2/bin/activate

echo "Testing model resnet-50 Q8 Model"
ONEDNN_MAX_CPU_ISA=AVX10_1_512 taskset -c 0    benchmark_app -m ./model/resnet-50-tf/FP32-INT8/resnet-50-tf.xml -b 1 -t 2 -hint latency -report_type detailed_counters -report_folder ./vec/resnet50/thread1/q8/
# ONEDNN_MAX_CPU_ISA=AVX10_1_512 taskset -c 0-3  benchmark_app -m ./model/resnet-50-tf/FP32-INT8/resnet-50-tf.xml -b 4 -t 2 -hint latency -report_type detailed_counters -report_folder ./vec/resnet50/thread4/q8/

echo "Testing model resnet-50 FP16 Model"
ONEDNN_MAX_CPU_ISA=AVX10_1_512 taskset -c 0    benchmark_app -m ./model/resnet-50-tf/FP16/resnet-50-tf.xml -b 1 -t 2 -hint latency -report_type detailed_counters -report_folder ./vec/resnet50/thread1/q16/
# ONEDNN_MAX_CPU_ISA=AVX10_1_512 taskset -c 0-3  benchmark_app -m ./model/resnet-50-tf/FP16/resnet-50-tf.xml -b 4 -t 2 -hint latency -report_type detailed_counters -report_folder ./vec/resnet50/thread4/q16/

echo "Testing model bert Q8 Model"
ONEDNN_MAX_CPU_ISA=AVX10_1_512 taskset -c 0    benchmark_app -m ./model/bert-base-ner/FP32-INT8/bert-base-ner.xml -b 1 -t 2 -hint latency -report_type detailed_counters -report_folder ./vec/bert/thread1/q8/
# ONEDNN_MAX_CPU_ISA=AVX10_1_512 taskset -c 0-3  benchmark_app -m ./model/bert-base-ner/FP32-INT8/bert-base-ner.xml -b 4 -t 2 -hint latency -report_type detailed_counters -report_folder ./vec/bert/thread4/q8/

echo "Testing model bert FP16 Model"
ONEDNN_MAX_CPU_ISA=AVX10_1_512 taskset -c 0    benchmark_app -m ./model/bert-base-ner/FP16/bert-base-ner.xml -b 1 -t 2 -hint latency -report_type detailed_counters -report_folder ./vec/bert/thread1/q16/
# ONEDNN_MAX_CPU_ISA=AVX10_1_512 taskset -c 0-3  benchmark_app -m ./model/bert-base-ner/FP16/bert-base-ner.xml -b 4 -t 2 -hint latency -report_type detailed_counters -report_folder ./vec/bert/thread4/q16/

echo "Testing model LLama3.2-1B WeightOnly-Q8-A16W8 Model"
source ../ov-llm-bench-env/bin/activate

ONEDNN_MAX_CPU_ISA=AVX10_1_512 taskset -c 0 python testllama.py --model ./model/llama.model/openvino/Llama-3.2-1B-Instruct-Q8-all/openvino_model.xml --batch_size 1 --output ./vec/llama/thread1/q8/llama_q8_thread1.csv
# ONEDNN_MAX_CPU_ISA=AVX10_1_512 taskset -c 0 python testllama.py --model ./model/llama.model/openvino/Llama-3.2-1B-Instruct-Q16/openvino_model.xml    --batch_size 1 --output ./vec/llama/thread1/q16/llama_q16_thread1.csv

# ONEDNN_MAX_CPU_ISA=AVX10_1_512 taskset -c 0-3 python testllama.py --model ./model/llama.model/openvino/Llama-3.2-1B-Instruct-Q8-all/openvino_model.xml --batch_size 4 --output ./vec/llama/thread4/q8/llama_q8_thread4.csv
# ONEDNN_MAX_CPU_ISA=AVX10_1_512 taskset -c 0-3 python testllama.py --model ./model/llama.model/openvino/Llama-3.2-1B-Instruct-Q16/openvino_model.xml    --batch_size 4 --output ./vec/llama/thread4/q16/llama_q16_thread4.csv

