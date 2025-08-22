import numpy as np
import os

# -----------------------------
# 配置参数
# -----------------------------
out_dir = "./llama_inputs"
batch_size = 1
seq_len = 128

# 创建输出目录
os.makedirs(out_dir, exist_ok=True)

# -----------------------------
# 构造输入张量
# -----------------------------
# 1. input_ids: 随机 token id
input_ids = np.random.randint(0, 1000, size=(batch_size, seq_len), dtype=np.int64)

# 2. attention_mask: 全1表示有效
attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

# 3. position_ids: 每个序列从0到seq_len-1
position_ids = np.tile(np.arange(seq_len, dtype=np.int64), (batch_size, 1))

# 4. beam_idx: 合法 KV cache 索引,[0,0,0,0]
beam_idx = np.zeros(batch_size, dtype=np.int32)  # [0,0,0,0]

# -----------------------------
# 保存为 .npy 文件
# -----------------------------
np.save(os.path.join(out_dir, "input_ids.npy"), input_ids)
np.save(os.path.join(out_dir, "attention_mask.npy"), attention_mask)
np.save(os.path.join(out_dir, "position_ids.npy"), position_ids)
np.save(os.path.join(out_dir, "beam_idx.npy"), beam_idx)

# -----------------------------
# 打印输出方便检查
# -----------------------------
print("Generated input tensors saved in:", out_dir)
print("input_ids shape:", input_ids.shape)
print("attention_mask shape:", attention_mask.shape)
print("position_ids shape:", position_ids.shape)
print("beam_idx shape:", beam_idx.shape)
print("beam_idx values:", beam_idx)
