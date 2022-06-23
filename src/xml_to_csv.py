import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            filename = root.find('filename').text
            image_size = root.find('size')
            width = float(image_size.find('width').text)
            height = float(image_size.find('height').text)
            class_name = member.find('name').text
            bndbox = member.find('bndbox')
            x_min = float(bndbox.find('xmin').text)
            y_min = float(bndbox.find('ymin').text)
            x_max = float(bndbox.find('xmax').text)
            y_max = float(bndbox.find('ymax').text)

            value = (filename, width, height, class_name, x_min, y_min, x_max, y_max)
            xml_list.append(value)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    print('Successfully converted xml to csv.')
    return xml_df



