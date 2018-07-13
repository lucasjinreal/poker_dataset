"""
convert the label data into tfrecord format
"""
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import tensorflow as tf

cls_txt_f = './poker_classes.txt'


# class label token from poker_classes.txt which DON'T need certain order
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x, y, w, h]


def create_img_raw(img_f):
    """
    convert a image to bytes, and resize to (416,416).
    be caution, this should be done for Yolo-V3
    :param img_f:
    :return:
    """
    img = Image.open(img_f)
    img_data = np.array(img, dtype='float32')/255
    return img_data.tobytes()


def parse_label(label_f, classes_list):
    tree = ET.parse(label_f)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    all_boxes = []
    for i, obj in enumerate(root.iter('object')):
        if i > 29:
            break
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes_list or int(difficult) == 1:
            continue
        cls_id = classes_list.index(cls)
        xml_box = obj.find('bndbox')
        b = (float(xml_box.find('xmin').text), float(xml_box.find('xmax').text), float(
            xml_box.find('ymin').text), float(xml_box.find('ymax').text))
        bb = convert((w, h), b) + [cls_id]
        all_boxes.extend(bb)
    if len(all_boxes) < 30 * 5:
        all_boxes = all_boxes + [0, 0, 0, 0, 0] * (30 - int(len(all_boxes) / 5))
    return np.array(all_boxes, dtype=np.float32).flatten().tolist()


def create_tf_record(img_label_dir):
    if os.path.exists(cls_txt_f):
        classes_list = open(cls_txt_f).readlines()
        print('Find all {} classes.'.format(len(classes_list)))

        if os.path.exists(img_label_dir):
            all_images = [os.path.join(img_label_dir, i) for i in os.listdir(img_label_dir)
                          if i.endswith('.jpg') or i.endswith('.png')
                          or i.endswith('.jpeg')]
            all_labels = [str(os.path.splitext(i)[0]) + '.xml' for i in all_images]

            writer = tf.python_io.TFRecordWriter('./train_data.tfrecord')

            for idx in range(len(all_images)):
                if os.path.exists(all_images[idx]):
                    if os.path.exists(all_labels[idx]):
                        img_f = all_images[idx]
                        label_f = all_labels[idx]

                        # start processing img and label
                        img_raw = create_img_raw(img_f)
                        xywhc = parse_label(label_f, classes_list)

                        example = tf.train.Example(features=tf.train.Features(feature={
                            'xywhc':
                                tf.train.Feature(float_list=tf.train.FloatList(value=xywhc)),
                            'img':
                                tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                        }))
                        writer.write(example.SerializeToString())
                    else:
                        print('image {} according label not exit, passing it.'.format(all_images[idx]))
                else:
                    print('image {} not exit, passing it.'.format(all_images[idx]))
            print('Done!')
        else:
            print('{} should contains images and labels but not exist.'.format(img_label_dir))
    else:
        print('{} class txt file not exist.'.format(cls_txt_f))


def main():
    create_tf_record('./images')


if __name__ == '__main__':
    main()