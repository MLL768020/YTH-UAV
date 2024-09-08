import os
import json
import shutil

def process_data(json_path, images_folder, output_folder):
    # 读取 JSON 文件
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    for item in data:
        # 获取当前项的id和file_name
        image_id = item['id']
        file_name = item['file_name']

        # 修改json中的file_name
        item['file_name'] = str(image_id)

        # 构建原始和目标文件路径
        original_image_path = os.path.join(images_folder, file_name)
        new_image_path = os.path.join(output_folder, str(image_id))

        # 修改图片的文件名并保存到新目录
        new_image_path_with_extension = os.path.join(output_folder, str(image_id) + os.path.splitext(file_name)[1])
        shutil.copy(original_image_path, new_image_path_with_extension)

    # 将修改后的JSON保存回文件
    output_json_path = os.path.join(output_folder, 'val.json')
    with open(output_json_path, 'w') as output_json_file:
        json.dump(data, output_json_file, indent=2)

if __name__ == "__main__":
    # 输入文件路径
    json_path = '/home/liyihang/miji/uav_palm_data1/annotations/instances_train2017.json'
    images_folder = '/home/liyihang/miji/uav_palm_data/train/images/'
    output_folder = '/home/liyihang/miji/uav_palm_data/val1/'

    # 处理数据
    process_data(json_path, images_folder, output_folder)
