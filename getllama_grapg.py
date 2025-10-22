import openvino as ov
import numpy as np

# ========== 1. 初始化 Core ==========
core = ov.Core()

# ========== 2. 读取并编译模型 ==========
model_path = "/home/perftest/perftool/openvino_report/model/llama.model/openvino/Llama-3.2-1B-Instruct-Q8/openvino_model.xml"
model = core.read_model(model_path)
compiled_model = core.compile_model(model, "CPU", config={"PERF_COUNT": "YES"})

# ========== 3. 准备输入 ==========
batch_size = 1
seq_len = 256

input_ids = np.random.randint(0, 1, size=(batch_size, seq_len), dtype=np.int64)
attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
beam_idx = np.array(range(batch_size), dtype=np.int32)

inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "position_ids": position_ids,
    "beam_idx": beam_idx,
}

# ========== 4. 创建推理请求 ==========
infer_request = compiled_model.create_infer_request()
infer_request.infer(inputs)

# ========== 5. 输出执行计划（Profiling） ==========
profiling_info = infer_request.get_profiling_info()

print("=== Execution Profiling Info ===")
for node in profiling_info:
    print(f"{node.node_name:60} | {node.status} | {node.real_time} ms")
