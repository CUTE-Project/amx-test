
echo "Testing Matmul MNK=256 --> 8192"

taskset -c 0 ./oneDNN/build/tests/benchdnn/benchdnn --mode=p --matmul --dt=u8:s8:s32 --wtag=any 256x256:256x256 512x512:512x512 768x768:768x768 1024x1024:1024x1024 1280x1280:1280x1280 1536x1536:1536x1536 1792x1792:1792x1792 2048x2048:2048x2048 2304x2304:2304x2304 2560x2560:2560x2560 2816x2816:2816x2816 3072x3072:3072x3072 3328x3328:3328x3328 3584x3584:3584x3584 3840x3840:3840x3840 4096x4096:4096x4096 4352x4352:4352x4352 4608x4608:4608x4608 4864x4864:4864x4864 5120x5120:5120x5120 5376x5376:5376x5376 5632x5632:5632x5632 5888x5888:5888x5888 6144x6144:6144x6144 6400x6400:6400x6400 6656x6656:6656x6656 6912x6912:6912x6912 7168x7168:7168x7168 7424x7424:7424x7424 7680x7680:7680x7680 7936x7936:7936x7936 8192x8192:8192x8192  > ./matmul/thread1/amx_matmul.csv

taskset -c 0-3 ./oneDNN/build/tests/benchdnn/benchdnn --mode=p --matmul --dt=u8:s8:s32 --wtag=any 256x256:256x256 512x512:512x512 768x768:768x768 1024x1024:1024x1024 1280x1280:1280x1280 1536x1536:1536x1536 1792x1792:1792x1792 2048x2048:2048x2048 2304x2304:2304x2304 2560x2560:2560x2560 2816x2816:2816x2816 3072x3072:3072x3072 3328x3328:3328x3328 3584x3584:3584x3584 3840x3840:3840x3840 4096x4096:4096x4096 4352x4352:4352x4352 4608x4608:4608x4608 4864x4864:4864x4864 5120x5120:5120x5120 5376x5376:5376x5376 5632x5632:5632x5632 5888x5888:5888x5888 6144x6144:6144x6144 6400x6400:6400x6400 6656x6656:6656x6656 6912x6912:6912x6912 7168x7168:7168x7168 7424x7424:7424x7424 7680x7680:7680x7680 7936x7936:7936x7936 8192x8192:8192x8192  > ./matmul/thread4/amx_matmul.csv

source ./openvino_env_2025.2/bin/activate

echo "Testing model resnet-50 Q8 Model"
taskset -c 0    benchmark_app -m ./model/resnet-50-tf/FP16-INT8/resnet-50-tf.xml -b 1 -t 2 -hint latency -report_type detailed_counters -report_folder ./resnet50/thread1/q8/
taskset -c 0-3  benchmark_app -m ./model/resnet-50-tf/FP16-INT8/resnet-50-tf.xml -b 4 -t 2 -hint latency -report_type detailed_counters -report_folder ./resnet50/thread4/q8/

echo "Testing model resnet-50 FP16 Model"
taskset -c 0    benchmark_app -m ./model/resnet-50-tf/FP16/resnet-50-tf.xml -b 1 -t 2 -hint latency -report_type detailed_counters -report_folder ./resnet50/thread1/q16/
taskset -c 0-3  benchmark_app -m ./model/resnet-50-tf/FP16/resnet-50-tf.xml -b 4 -t 2 -hint latency -report_type detailed_counters -report_folder ./resnet50/thread4/q16/

echo "Testing model bert Q8 Model"
taskset -c 0    benchmark_app -m ./model/bert-base-ner/FP32-INT8/bert-base-ner.xml -b 1 -t 2 -hint latency -report_type detailed_counters -report_folder ./bert/thread1/q8/
taskset -c 0-3  benchmark_app -m ./model/bert-base-ner/FP32-INT8/bert-base-ner.xml -b 4 -t 2 -hint latency -report_type detailed_counters -report_folder ./bert/thread4/q8/

echo "Testing model bert FP16 Model"
taskset -c 0    benchmark_app -m ./model/bert-base-ner/FP16/bert-base-ner.xml -b 1 -t 2 -hint latency -report_type detailed_counters -report_folder ./bert/thread1/q16/
taskset -c 0-3  benchmark_app -m ./model/bert-base-ner/FP16/bert-base-ner.xml -b 4 -t 2 -hint latency -report_type detailed_counters -report_folder ./bert/thread4/q16/

echo "Testing model LLama3.2-1B WeightOnly-Q8-A16W8 Model"
source ./ov-llm-bench-env/bin/activate

ONEDNN_VERBOSE=1 taskset -c 0 python ./openvino.genai/tools/llm_bench/benchmark.py -p 'a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a ' -bs 1 -ic 1 -m /home/perftest/perftool/openvino_report/model/llama.model/openvino/Llama-3.2-1B-Instruct-Q8 > ./llama/thread1/W8A16/benchmark_detailed_counters_report.csv

ONEDNN_VERBOSE=1 taskset -c 0-4 python ./openvino.genai/tools/llm_bench/benchmark.py -p 'a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a ' -bs 4 -ic 1 -m /home/perftest/perftool/openvino_report/model/llama.model/openvino/Llama-3.2-1B-Instruct-Q8 > ./llama/thread4/W8A16/benchmark_detailed_counters_report.csv

