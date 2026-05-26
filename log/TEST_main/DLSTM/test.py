def count_ones_ratio(file_path):
    total = 0
    ones = 0

    with open(file_path, 'r') as file:
        for line in file:
            for char in line.strip():
                if char in ['0', '1']:
                    total += 1
                    if char == '1':
                        ones += 1

    if total == 0:
        return 0.0
    return ones / total


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("用法: python count_ones_ratio.py 文件路径")
    else:
        file_path = sys.argv[1]
        ratio = count_ones_ratio(file_path)
        print(f"1 的比例为: {ratio:.4f}")