import json
import cv2
import os

all_classes = {'Dead': '0', 'Health': '1', 'Grass': '2', 'Small': '3', 'Yellow': '4'}  ##类别列表，与训练配置文件中的顺序保持一致
savepath = "/home/liyihang/miji/uav_palm_data/train/labels/"  # txt文件存放位置
jsonpath = '/home/liyihang/miji/uav_palm_data/train/annotations/' # json文件位置
imgpath = "/home/liyihang/miji/uav_palm_data/train/images/"  # 图片位置，因为我的json文件中没有图片size，故需要读取图片得到size
json_files = os.listdir(jsonpath)
for i in json_files:
    infile = jsonpath + i
    with open(infile, 'r') as load_f:
        load_dict = json.load(load_f)  # 打开每个json文件

    outfile = open(savepath + load_dict["imagePath"][:-4] + '.txt', 'w')

    img_path = imgpath + load_dict["imagePath"]
    img = cv2.imread(img_path)
    size = img.shape
    h_img, w_img = size[0], size[1]  # 得到图片size

    for item in load_dict["shape"]:
        print(item)
        label_int = all_classes[item['label']]
        print(infile)
        if not item['boxes']:
            continue
        x1, y1, x2, y2 = item['boxes']
        print(label_int)
        print(x1, y1, x2, y2)
        x_center = (x1 + x2) / 2 / w_img
        y_center = (y1 + y2) / 2 / h_img
        w = (x2 - x1) / w_img
        h = (y2 - y1) / h_img
        outfile.write(str(label_int) + " " + str(x_center) + " " + str(y_center) + " " + str(w) + " " + str(h) + '\n')
    outfile.close()


