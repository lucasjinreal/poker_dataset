# -*- coding: utf-8 -*-
# file: xml_to_csv.py
# author: JinTian
# time: 2018/6/13 6:37 PM
# Copyright 2018 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""
convert xml label generate by labelImg into csv format

the result like this:

image_file.jpg x,x,x,x x,x,x,x ....

"""
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path, class_f):

    curr_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
    if os.path.exists(class_f):
        with open(class_f, 'r') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
    else:
        raise ValueError('Class file should provide.')

    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        value = os.path.join(curr_dir, root.find('filename').text) + ' '
        print(value)
        for member in root.findall('object'):
            value += '{},{},{},{},{} '.format(
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text),
                class_names.index(member[0].text)
            )
        xml_list.append(value + '\n')
    with open('train.txt', 'w+') as f:
        f.writelines(xml_list)
    print('Successfully converted xml to txt.')


def main():
    xml_to_csv('./images', 'poker_classes.txt')


if __name__ == '__main__':
    main()
