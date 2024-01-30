# # srcファイルのタイトル全てにzfillを適用する
# import re

# file_path = './Dataset/src/BrandSilence.txt'  # ファイルのパスを指定

# with open(file_path, 'r') as file:
#     lines = file.readlines()

# # BrandSilenceで始まる行を探し、直後の数字を取得してintに変換する
# for i, line in enumerate(lines):
#     if line.startswith("BrandSilence"):
#         # 正規表現を使用してBrandSilenceの直後の数字を抽出
#         match = re.search(r'BrandSilence(\d+)', line)
#         if match:
#             # 抽出した数字をintに変換
#             extracted_number = match.group(1)
            
#             # ここで extracted_number を利用して何らかの処理を行う例
#             # 例: extracted_number を2倍にして元の行に置き換える
#             new_line = line.replace(f"BrandSilence{extracted_number}", f"BrandSilence{extracted_number.zfill(5)}")
#             lines[i] = new_line

# # 変更後の内容をファイルに書き込む
# with open(file_path, 'w') as file:
#     file.writelines(lines)


# # bagファイルの名前を変更する
# import re
# import os

# # ファイルが存在するディレクトリのパス
# directory_path = './Dataset/rosbag/'

# # ディレクトリ内のファイルをリストアップ
# files = os.listdir(directory_path)

# # BrandSilence{num}.bag のファイルに対して処理
# for file_name in files:
#     if file_name.startswith("BrandSilence") and file_name.endswith(".bag"):
#         # 正規表現を使用して num を抽出
#         match = re.search(r'BrandSilence(\d+)\.bag', file_name)
#         if match:
#             num = match.group(1)
#             # 0埋めした数字を含む新しいファイル名を生成
#             new_file_name = f"BrandSilence{num.zfill(5)}.bag"
            
#             # ファイルのリネーム
#             old_path = os.path.join(directory_path, file_name)
#             new_path = os.path.join(directory_path, new_file_name)
#             os.rename(old_path, new_path)
#             print(f"Renamed: {file_name} to {new_file_name}")
