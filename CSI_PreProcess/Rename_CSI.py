# 由于原始数据在不同文件夹中类别的分属不同，因此将原始信号的标签进行重新统计，整理为22个编号，表示22种手势
# 同时统计每个用户/类别的数据量

import os
import re
import shutil

room_map = {
    '20181109': 1,
    '20181112': 1,
    '20181116': 1,
    '20181115': 1,
    '20181117': 2,
    '20181118': 2,
    '20181121': 1,
    '20181127': 2,
    '20181205_user3': 2,
    '20181128': 2,
    '20181130': 1,
    '20181204': 2,
    '20181205_user2': 2,
    '20181208_user2': 2,
    '20181208_user3': 2,
    '20181209_user2': 2,
    '20181209_user6': 2,
    '20181211': 3
}

label_map = {
    '20181109': {5: 10, 6: 11},
    '20181112': {1: 13, 2: 14, 3: 15, 4: 16, 5: 17, 6: 18, 7: 19, 8: 20, 9: 21, 0: 22},
    '20181116': {1: 13, 2: 14, 3: 15, 4: 16, 5: 17, 6: 18, 7: 19, 8: 20, 9: 21, 0: 22},
    '20181115': {4: 12, 5: 10, 6: 11},
    '20181117': {4: 12, 5: 10, 6: 11},
    '20181118': {4: 12, 5: 10, 6: 11},
    '20181121': {2: 6, 3: 9, 4: 5, 5: 8, 6: 7, 1: 4},
    '20181127': {2: 6, 3: 9, 4: 5, 5: 8, 6: 7, 1: 4},
    '20181205_user3': {2: 6, 3: 9, 4: 5, 5: 8, 6: 7, 1: 4},
    '20181128': {4: 6, 5: 9, 6: 5},
    '20181130': {5: 6, 6: 9, 7: 5, 9: 7},
    '20181204': {5: 6, 6: 9, 7: 5, 9: 7},
    '20181205_user2': {1: 6, 2: 9, 3: 5, 4: 8, 5: 7},
    '20181208_user2': {},
    '20181208_user3': {},
    '20181209_user2': {},
    '20181209_user6': {5: 6, 6: 9},
    '20181211': {5: 6, 6: 9}
}

empty_files = ['20181109/user2/user2-6-4-4-2-r1.dat',
                '20181109/user3/user3-1-3-1-8-r5.dat',
                '20181118/user2/user2-3-5-3-4-r4.dat',
                '20181209_user6/user6-3-1-1-5-r5.dat',
                '20181211/user8/user8-1-1-1-1-r5.dat',
                '20181211/user8/user8-3-3-3-5-r2.dat',
                '20181211/user9/user9-1-1-1-1-r1.dat']

# 只要 empty_files 中出现某个样本的任一 receiver，
# 就将该样本的所有 receiver 文件全部排除。
empty_sample_prefixes = {
    re.sub(r'-r\d+\.dat$', '', p.replace('\\', '/'))
    for p in empty_files
}

src_folder = 'CSI_raw'

dst_folder = 'CSI'

# 运行前是否清空目标目录内容（仅清空目录内文件，不删除目录本身）
# True: 每次运行先清空 dst_folder，避免历史文件造成重名
# False: 保留历史结果
clear_dst_before_run = False

# 1. 遍历文件，提取出其中的信息：user_id-label-torso_loc-face_dir-rep_num-receiver_id
# 2. 根据label_map进行标签的重新统计，得到新的标签编号和room_num
# 3. 同时统计每个用户/类别的数据量，输出到dst_folder/data_count.txt中
# src_folder: CSI_raw, dst_folder: CSI
def rename_dir(src_folder, dst_folder, folder_num):
    count_by_user_label = {}
    sample_seen = set()
    excluded_files = []
    bad_format_files = []
    conflict_files = []
    rename_records = []
    cur_folder_path = os.path.join(src_folder, folder_num)
    for root, dirs, files in os.walk(cur_folder_path):
        for file in files:
            if file.endswith('.dat'):
                src_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(src_file_path, src_folder)
                relative_path = relative_path.replace('\\', '/')
                sample_prefix = re.sub(r'-r\d+\.dat$', '', relative_path)
                if sample_prefix in empty_sample_prefixes:
                    print(f"跳过空白样本(全部 receiver): {relative_path}")
                    excluded_files.append(relative_path)
                    continue
                parts = file.split('-')
                if len(parts) < 6:
                    print(f"文件名格式不正确: {file}")
                    bad_format_files.append(relative_path)
                    continue
                user_id = parts[0]
                label = int(parts[1])
                torso_loc = parts[2]
                face_dir = parts[3]
                rep_num = parts[4]
                receiver_id = parts[5].split('.')[0]

                if not rep_num.isdigit():
                    print(f"rep_num 格式不正确: {file}")
                    bad_format_files.append(relative_path)
                    continue

                if folder_num not in room_map:
                    raise KeyError(f"room_map 中未找到 folder_num: {folder_num}")
                room_num = room_map[folder_num]

                if folder_num not in label_map:
                    raise KeyError(f"label_map 中未找到 folder_num={folder_num}")
                new_label = label_map[folder_num].get(label, label)

                rep_num_new = int(rep_num)
                while True:
                    new_file_name = f"{user_id}-{new_label}-{torso_loc}-{face_dir}-{rep_num_new}-{receiver_id}-room{room_num}.dat"
                    dst_file_path = os.path.join(dst_folder, new_file_name)
                    if not os.path.exists(dst_file_path):
                        break
                    rep_num_new += 1

                if rep_num_new != int(rep_num):
                    print(f"检测到重名，rep_num 自增: {rep_num} -> {rep_num_new} ({relative_path})")

                count_key = (user_id, new_label)
                count_by_user_label[count_key] = count_by_user_label.get(count_key, 0) + 1

                # 一个动作样本包含多个 receiver，样本级统计不区分 receiver_id
                sample_key = (user_id, new_label, torso_loc, face_dir, rep_num_new, room_num)
                sample_seen.add(sample_key)

                os.rename(src_file_path, dst_file_path)
                print(f"Renaming: {src_file_path} -> {dst_file_path}")
                rename_records.append((relative_path, new_file_name))

    return count_by_user_label, sample_seen, excluded_files, bad_format_files, conflict_files, rename_records


def write_data_count(dst_folder, count_by_user_label, sample_seen, excluded_files, bad_format_files, conflict_files):
    count_file = os.path.join(dst_folder, 'data_count.txt')

    user_totals = {}
    label_totals = {}
    for (user_id, label), cnt in count_by_user_label.items():
        user_totals[user_id] = user_totals.get(user_id, 0) + cnt
        label_totals[label] = label_totals.get(label, 0) + cnt

    with open(count_file, 'w', encoding='utf-8') as f:
        f.write('Widar3.0 Data Count Summary\n')
        f.write('==========================\n')
        f.write(f'Total .dat files: {sum(count_by_user_label.values())}\n')
        f.write(f'Total unique samples (ignore receiver): {len(sample_seen)}\n\n')

        f.write('Excluded Files (matched empty sample prefixes)\n')
        f.write('--------------------------------------------\n')
        f.write(f'Total excluded files: {len(excluded_files)}\n')
        for p in sorted(excluded_files):
            f.write(f'{p}\n')

        f.write('\nBad Format Files (filename split parts < 6)\n')
        f.write('------------------------------------------\n')
        f.write(f'Total bad format files: {len(bad_format_files)}\n')
        for p in sorted(bad_format_files):
            f.write(f'{p}\n')

        f.write('\nConflict Files (destination already exists)\n')
        f.write('------------------------------------------\n')
        f.write(f'Total conflict files: {len(conflict_files)}\n')
        for src_rel, dst_name in sorted(conflict_files):
            f.write(f'{src_rel}\t->\t{dst_name}\n')

        f.write('\n')

        f.write('Per User / Label (.dat file count)\n')
        f.write('----------------------------------\n')
        for user_id, label in sorted(count_by_user_label.keys(), key=lambda x: (x[0], x[1])):
            f.write(f'{user_id}\tlabel={label}\tcount={count_by_user_label[(user_id, label)]}\n')

        f.write('\nPer User Total (.dat file count)\n')
        f.write('-------------------------------\n')
        for user_id in sorted(user_totals.keys()):
            f.write(f'{user_id}\ttotal={user_totals[user_id]}\n')

        f.write('\nPer Label Total (.dat file count)\n')
        f.write('--------------------------------\n')
        for label in sorted(label_totals.keys()):
            f.write(f'label={label}\ttotal={label_totals[label]}\n')

    print(f'数据统计已写入: {count_file}')


def write_rename_audit(dst_folder, rename_records):
    audit_file = os.path.join(dst_folder, 'rename_audit.txt')
    with open(audit_file, 'w', encoding='utf-8') as f:
        f.write('Widar3.0 Rename Audit\n')
        f.write('====================\n')
        f.write(f'Total renamed files: {len(rename_records)}\n\n')
        for src_rel, dst_name in rename_records:
            f.write(f'{src_rel}\t->\t{dst_name}\n')

    print(f'重命名审计已写入: {audit_file}')


def clear_directory_contents(dir_path):
    if not os.path.exists(dir_path):
        return

    for name in os.listdir(dir_path):
        p = os.path.join(dir_path, name)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
        except Exception as err:
            print(f"清理失败: {p}, 错误: {err}")

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dst_folder_path = os.path.join(root_dir, dst_folder)
    src_folder_path = os.path.join(root_dir, src_folder)
    if not os.path.exists(dst_folder_path):
        os.makedirs(dst_folder_path)

    if clear_dst_before_run:
        print(f"运行前清空目标目录: {dst_folder_path}")
        clear_directory_contents(dst_folder_path)

    subfolders = []
    all_count_by_user_label = {}
    all_sample_seen = set()
    all_excluded_files = []
    all_bad_format_files = []
    all_conflict_files = []
    all_rename_records = []

    # 遍历根目录下的所有子文件夹，格式为20181109、20181112等
    for d in os.listdir(src_folder_path):
        subfolder_path = os.path.join(src_folder_path, d)
        if os.path.isdir(subfolder_path):
            subfolders.append(d)
            print(f"Processing folder: {d}")
            folder_counts, folder_samples, excluded_files, bad_format_files, conflict_files, rename_records = rename_dir(src_folder_path, dst_folder_path, d)

            for key, cnt in folder_counts.items():
                all_count_by_user_label[key] = all_count_by_user_label.get(key, 0) + cnt
            all_sample_seen.update(folder_samples)
            all_excluded_files.extend(excluded_files)
            all_bad_format_files.extend(bad_format_files)
            all_conflict_files.extend(conflict_files)
            all_rename_records.extend(rename_records)

    for folder in sorted(subfolders):
        print(folder)

    write_data_count(
        dst_folder_path,
        all_count_by_user_label,
        all_sample_seen,
        all_excluded_files,
        all_bad_format_files,
        all_conflict_files,
    )
    write_rename_audit(dst_folder_path, all_rename_records)


if __name__ == '__main__':
    main()
    