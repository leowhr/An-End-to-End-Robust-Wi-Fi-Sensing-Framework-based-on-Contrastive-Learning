import os
import shutil

src_dir = "./dataset/10fenlei"
dst_dir = "./dataset/10fenlei(test)"

os.makedirs(dst_dir, exist_ok=True)

valid_numbers = {"6", "7", "8"}

for filename in os.listdir(src_dir):
    src_path = os.path.join(src_dir, filename)

    if not os.path.isfile(src_path):
        continue

    parts = filename.split('-')

    # 至少要有3段
    if len(parts) < 3:
        continue

    third_part = parts[2]  # 第三个字段

    if third_part in valid_numbers:
        dst_path = os.path.join(dst_dir, filename)
        shutil.copy2(src_path, dst_path)

print("完成筛选复制！")