import re
from collections import defaultdict
from multiprocessing import Pool

# 1. 定义29条正则匹配规则
patterns = [
    r"Adding an already existing block (.*)",
    r"(.*)Verification succeeded for (.*)",
    r"(.*) Served block (.*) to (.*)",
    r"(.*):Got exception while serving (.*) to (.*):(.*)",
    r"Receiving block (.*) src: (.*) dest: (.*)",
    r"Received block (.*) src: (.*) dest: (.*) of size ([-]?[0-9]+)",
    r"writeBlock (.*) received exception (.*)",
    r"PacketResponder ([-]?[0-9]+) for block (.*) Interrupted\.",
    r"Received block (.*) of size ([-]?[0-9]+) from (.*)",
    r"PacketResponder (.*) ([-]?[0-9]+) Exception (.*)",
    r"PacketResponder ([-]?[0-9]+) for block (.*) terminating",
    r"(.*):Exception writing block (.*) to mirror (.*)(.*)",
    r"Receiving empty packet for block (.*)",
    r"Exception in receiveBlock for block (.*) (.*)",
    r"Changing block file offset of block (.*) from ([-]?[0-9]+) to ([-]?[0-9]+) meta file offset to ([-]?[0-9]+)",
    r"(.*):Transmitted block (.*) to (.*)",
    r"(.*):Failed to transfer (.*) to (.*) got (.*)",
    r"(.*) Starting thread to transfer block (.*) to (.*)",
    r"Reopen Block (.*)",
    r"Unexpected error trying to delete block (.*)\. BlockInfo not found in volumeMap\.",
    r"Deleting block (.*) file (.*)",
    r"BLOCK\* NameSystem\.allocateBlock: (.*)\. (.*)",
    r"BLOCK\* NameSystem\.delete: (.*) is added to invalidSet of (.*)",
    r"BLOCK\* Removing block (.*) from neededReplications as it does not belong to any file\.",
    r"BLOCK\* ask (.*) to replicate (.*) to (.*)",
    r"BLOCK\* NameSystem\.addStoredBlock: blockMap updated: (.*) is added to (.*) size ([-]?[0-9]+)",
    r"BLOCK\* NameSystem\.addStoredBlock: Redundant addStoredBlock request received for (.*) on (.*) size ([-]?[0-9]+)",
    r"BLOCK\* NameSystem\.addStoredBlock: addStoredBlock request received for (.*) on (.*) size ([-]?[0-9]+) But it does not belong to any file\.",
    r"PendingReplicationMonitor timed out block (.*)"
]


# 2. 日志处理函数
def extract_block_id(line):
    """从日志中提取 block_id（blk_ 开头的）"""
    match = re.search(r"(blk_[\-0-9]+)", line)
    return match.group(1) if match else None


def process_log_chunk(log_chunk):
    """处理日志块，提取匹配到的正则表达式并统计"""
    block_vectors = defaultdict(lambda: [0] * len(patterns))

    for line in log_chunk:
        block_id = extract_block_id(line)
        if not block_id:
            continue

        for i, pattern in enumerate(patterns):
            if re.search(pattern, line):
                block_vectors[block_id][i] += 1

    # 需要转换为可序列化的普通 dict
    return dict(block_vectors)


def process_logs(log_file_path, chunk_size=1000):
    """读取日志文件并并行处理日志块"""
    block_vectors = defaultdict(lambda: [0] * len(patterns))
    with open(log_file_path, 'r') as f:
        lines = f.readlines()

    # 使用多进程处理日志块
    with Pool() as pool:
        chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
        results = pool.map(process_log_chunk, chunks)

    # 合并结果
    for result in results:
        for block_id, vector in result.items():
            block_vectors[block_id] = [sum(x) for x in zip(block_vectors[block_id], vector)]

    return block_vectors


# 3. 输出向量矩阵
def save_vectors_to_file(vectors, output_file_path):
    """保存向量矩阵到文件"""
    with open(output_file_path, 'w') as f:
        for block_id, vector in vectors.items():
            f.write(" ".join(map(str, vector)) + f"  %%{block_id}\n")


# 4. 输出日志索引
def save_log_index_to_file(vectors, output_file_path):
    """保存日志索引到文件"""
    with open(output_file_path, 'w') as f:
        for block_id in vectors:
            f.write(f"%%{block_id}\n")


# =============== 主程序入口 ====================
if __name__ == "__main__":
    log_file = "sorted.log.txt"  # 原始日志文件路径
    output_file1 = "rawTFVector.txt"  # 输出矩阵路径
    output_file2 = "log_index.txt"  # 输出日志索引

    # 处理日志并保存结果
    vectors = process_logs(log_file)
    save_vectors_to_file(vectors, output_file1)
    save_log_index_to_file(vectors, output_file2)

    print(f"共处理 {len(vectors)} 个 block，结果已保存到 {output_file1} 和 {output_file2}")
