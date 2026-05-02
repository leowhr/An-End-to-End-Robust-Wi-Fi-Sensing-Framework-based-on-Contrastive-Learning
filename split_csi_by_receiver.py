import os
from scipy.io import loadmat, savemat

# === 输入输出目录 ===
data_dir = "/mnt/data/keran/LKR/WiSRL/dataset/10fenlei"
save_dir = "/mnt/data/keran/LKR/LLM-GEN/RF-Painter/dataset/10fenlei"

os.makedirs(save_dir, exist_ok=True)

for filename in os.listdir(data_dir):
    if not filename.endswith(".mat"):
        continue

    filepath = os.path.join(data_dir, filename)
    data = loadmat(filepath)

    # 固定变量名为 CSI_result
    if "CSI_result" not in data:
        print(f"⚠️ 文件 {filename} 中找不到 'CSI_result'，跳过。")
        continue

    arr = data["CSI_result"]

    # 检查形状是否正确
    if arr.ndim != 3 or arr.shape[2] != 6:
        print(f"⚠️ 文件 {filename} 形状为 {arr.shape}，不是 (1450, 90, 6)，跳过。")
        continue

    # 拆成 6 份保存
    for i in range(6):
        part = arr[:, :, i]
        new_name = filename.replace(".mat", f"-r{i+1}.mat")
        new_path = os.path.join(save_dir, new_name)
        savemat(new_path, {"CSI_result": part})

    print(f"✅ 拆分完成：{filename}")

print("🎉 全部文件处理完成！")