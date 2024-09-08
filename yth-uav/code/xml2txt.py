import xml.etree.ElementTree as ET
from os import listdir, getcwd
import glob

# classes information
classes = ["stressed", "healthy"]

def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(image_name):
    print(f"Processing {image_name}")
    f = open('./Annotations/' + image_name[:-3] + 'xml', encoding="utf8") # xml file path
    out_file = open('./label/' + image_name[:-3] + 'txt', 'w')  # txt file path
    xml_text = f.read()
    root = ET.fromstring(xml_text)
    f.close()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            print(cls)
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

if __name__ == '__main__':

    for image_path in glob.glob("./JPEGImages/*.jpg"):  # image file path
        image_name = image_path.split('\\')[-1]
        convert_annotation(image_name)