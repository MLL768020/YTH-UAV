import json
from pathlib import Path

# 读取 precision.json 文件
with open('precision.json', 'r') as f:
    data = json.load(f)

# 获取图片名称并按照从小到大排序
image_names = sorted(set([item['image_id'] for item in data]))

# 构建一个映射关系，将图片名称映射为新的image_id（从0开始）
image_id_mapping = {image_name: str(i) for i, image_name in enumerate(image_names)}

# 更新数据中的image_id
for item in data:
    item['image_id'] = image_id_mapping[item['image_id']]

# 保存为新的JSON文件
new_json_filename = 'new_precision.json'
with open(new_json_filename, 'w') as f:
    json.dump(data, f)

print(f'文件已保存为 {new_json_filename}')
