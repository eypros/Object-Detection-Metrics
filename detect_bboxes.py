# coding=utf-8

__version__ = 0.1

import argparse
import os
import re
import time
from glob import glob
import xml.etree.ElementTree as ET
from xml.dom import minidom

import tensorflow as tf
import numpy as np
from PIL import Image


def get_arguments():
    """
    Parse the arguments
    :return:
    """
    parser = argparse.ArgumentParser(
        prog='Write object detection bboxes in files',
        description='This module applies a trained object detection model to a set of images in order to produce '
                    'bboxes.\n Bboxes are written either to xml or txt file and are used to evaluate the trained model',
        epilog="Developed by: George Orfanidis (g.orfanidis@it.gr)")
    # formatter_class=RawTextHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {}'.format(__version__))
    # Positional arguments
    # Mandatory
    parser.add_argument(
        '-i',
        '--image_path',
        help='folder containing the images to apply the detection model')
    parser.add_argument(
        '-m',
        '--model-path',
        help='folder containing the trained model path. This is the folder where the "frozen_inference_graph.pb" '
             'resides in other words')
    # Optional
    parser.add_argument(
        '-l',
        '--label-map-path',
        help='the path to the label map for this model.')
    parser.add_argument(
        '--score-thres',
        dest='score_thres',
        type=float,
        default=0.0,
        help='the threshold under which the bboxes will be ignored and not written to the output files.\n'
             'Default value is 0.0')
    parser.add_argument(
        '--accepted-classes',
        default='',
        help='A list with all classes to be taken into consideration when writing the bboxes in files.\n'
             'Default value is an empty list which corresponds to take into consideration all available classes')
    parser.add_argument(
        '-t'
        '--txt_file',
        dest='txt_file',
        action='store_true',
        help='Whether output file will be xml or txt.\n'
             'If not set xml is used (default).')

    args = parser.parse_args()

    return args


def create_output_files_from_detection(image_path, model_path, label_map_path, score_thres=0.0, accepted_classes=[],
                                       xml_file=True):
    """
    Apply an object detection model to a folder and writes donw in xml or txt format the detections (bboxes etc)
    :param accepted_classes:
    :param score_thres:
    :param image_path:
    :param model_path:
    :param label_map_path:
    :param xml_file:
    :return:
    """

    if not label_map_path:
        label_map_path = os.path.join(model_path, 'pipeline.config') if \
            os.path.exists(os.path.join(model_path, 'pipeline.config')) else os.path.join(os.path.dirname(
            os.path.dirname(os.path.join(model_path))), 'pipeline.config')
    categories_dict = get_label_map(label_map_path, name_to_id=False)

    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(os.path.join(model_path, 'frozen_inference_graph.pb'), 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    image_list = get_list_files(image_path, '*.jpg', relative=False)
    suffix = os.path.dirname(os.path.join(model_path, 'frozen_inference_graph.pb')).rsplit('snapshots_', 1)[1].rsplit(
        '_', 1)[0] if 'snapshots_' in model_path else 'detection'

    start_det = time.time()
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=config) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for im in image_list:
                print(im)
                image = Image.open(im)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # write_on = image_np.copy()
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes_det, scores_det, labels_det_ids, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                boxes_det, scores_det, labels_det_ids = filter_bboxes_by_score(boxes_det, scores_det, labels_det_ids,
                                                                               image_np, score_thres=score_thres,
                                                                               width_first=True)
                # h, w, _ = image_np.shape
                # boxes_det = np.squeeze(boxes_det)
                # boxes_det = np.array([boxes_det[:, 0] * h, boxes_det[:, 1] * w, boxes_det[:, 2] * h,
                #                       boxes_det[:, 3] * w]).T
                if xml_file:
                    create_single_xml_from_detection_result(image_path, 'annotations_{}'
                                                            .format(suffix),
                                                            os.path.basename(im),
                                                            image_np, boxes_det, scores_det,
                                                            labels_det_ids,
                                                            categories_dict,
                                                            accepted_classes=accepted_classes,
                                                            dataset=suffix)
                else:
                    filename = os.path.join(image_path, 'detections_txt_{}'.format(suffix),
                                            '{}.txt'.format(os.path.basename(im).rsplit('.', 1)[0]))
                    if not os.path.exists(os.path.join(image_path, 'detections_txt_{}'.format(suffix))):
                        os.mkdir(os.path.join(image_path, 'detections_txt_{}'.format(suffix)))
                    with open(filename, 'w') as f:
                        for i in range(boxes_det.shape[0]):
                            if accepted_classes and categories_dict[labels_det_ids[i]] not in accepted_classes:
                                continue
                            f.write('{} {:2.5f} {} {} {} {}\n'.format(
                                categories_dict[labels_det_ids[i]], scores_det[i], int(round(boxes_det[i, 0])),
                                int(round(boxes_det[i, 1])), int(round(boxes_det[i, 2])),
                                int(round(boxes_det[i, 3]))))

    end = time.time()
    print('Elapsed time detecting objects and writing corresponding bboxes: {:2.3f} secs'.format(end - start_det))


def load_image_into_numpy_array(image_input):
    """

    :param image_input:
    :return:
    """
    (im_width, im_height) = image_input.size
    return np.array(image_input.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def get_label_map(label_maps_path, name_to_id=True):
    """

    :param label_maps_path:
    :param name_to_id:
    :return:
    """
    label_map = {}
    # pattern = re.compile("\'(.+?)\'")  # re.compile("\'[a-zA-Z]\'")
    with open(label_maps_path, 'r') as f:
        for line in f:
            if line == 'item {' or line == '':
                continue
            else:
                if 'id' in line:
                    item_id = int(line.split('id: ')[1])
                elif 'name' in line:
                    # item_name = line.split('name: ')[1]
                    item_name = re.findall("\'(.+?)\'", line)[0]
                    # item_name = re.search(pattern, line)
                    # item_name = item_name.group(1)
                    if name_to_id:
                        label_map[item_name] = item_id
                    else:
                        label_map[item_id] = item_name
    return label_map


def get_list_files(folder_path, pattern='*', relative=True, extension=True):
    """
    Returns the list of files inside a folder
    :param folder_path:
    :param pattern:
    :param relative:
    :param extension:
    :return:
    """

    if not os.path.isdir(folder_path):
        if relative:
            if extension:
                return [os.path.basename(folder_path)]
            else:
                return [os.path.splitext(os.path.basename(folder_path))[0]]
        else:
            if extension:
                return [folder_path]
            else:
                return [os.path.splitext(folder_path)[0]]

    if relative:
        if extension:
            files_list = [os.path.relpath(x, folder_path) for x in glob(os.path.join(folder_path, pattern))]
        else:
            files_list = [os.path.splitext(os.path.relpath(x, folder_path))[0]
                          for x in glob(os.path.join(folder_path, pattern))]
    else:
        if extension:
            files_list = glob(os.path.join(folder_path, pattern))
        else:
            files_list = [os.path.splitext(x)[0] for x in glob(os.path.join(folder_path, pattern))]
    return files_list


def filter_bboxes_by_score(boxes_det, scores, classes, image_np, score_thres=0.5, width_first=True):
    """
     A simple function to keep only the bboxes (as well as the corresponding scores and labels) with score
    above the threshold
    :param width_first:
    :param boxes_det:
    :param scores:
    :param classes:
    :param image_np:
    :param score_thres:
    :return:
    """
    boxes_det = np.squeeze(boxes_det)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)
    h, w, _ = image_np.shape
    res = np.where(scores > score_thres)
    if not res[0].shape[0]:
        boxes_det = np.zeros((0, 4))
        scores = np.zeros((0, 1))
        classes = np.zeros((0, 1))
        return boxes_det, scores, classes
    n = np.where(scores > score_thres)[0][-1] + 1

    # this creates an array with just enough rows as object with score above the threshold
    if width_first:
        # format: absolute x, y, x, y
        boxes_det = np.array([boxes_det[:n, 1] * w, boxes_det[:n, 0] * h, boxes_det[:n, 3] * w, boxes_det[:n, 2] * h]).T
    else:
        boxes_det = np.array([boxes_det[:n, 0] * h, boxes_det[:n, 1] * w, boxes_det[:n, 2] * h, boxes_det[:n, 3] * w]).T
    classes = classes[:n]
    scores = scores[:n]
    return boxes_det, scores, classes


def create_single_xml_from_detection_result(image_folder, rel_anno_folder, image_name, image_np, bboxes, scores,
                                            labels, categories_dict, accepted_classes=[], dataset=''):
    """

    :param image_folder:
    :param rel_anno_folder:
    :param image_name:
    :param image_np:
    :param bboxes:
    :param scores:
    :param labels:
    :param categories_dict:
    :param accepted_classes:
    :param dataset:
    """
    annot = ET.Element('annotation')
    fol = ET.SubElement(annot, 'folder')
    fol.text = os.path.basename(os.path.join(image_folder))
    filename = ET.SubElement(annot, 'filename')
    filename.text = image_name
    path1 = ET.SubElement(annot, 'path')
    path1.text = os.path.join(image_folder, image_name)
    source = ET.SubElement(annot, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown' if not dataset else dataset
    size = ET.SubElement(annot, 'size')
    h, w, d = image_np.shape
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = str(d)
    segmented = ET.SubElement(annot, 'segmented')
    segmented.text = '0'
    for i in range(bboxes.shape[0]):
        if accepted_classes and categories_dict[labels[i]] not in accepted_classes:
            continue
        object1 = ET.SubElement(annot, 'object')
        name = ET.SubElement(object1, 'name')
        name.text = categories_dict[labels[i]]
        pose = ET.SubElement(object1, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(object1, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(object1, 'difficult')
        difficult.text = '0'
        score = ET.SubElement(object1, 'score')
        score.text = str(scores[i])
        bndbox = ET.SubElement(object1, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(int(round(bboxes[i, 0])))
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(int(round(bboxes[i, 1])))
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(int(round(bboxes[i, 2])))
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(int(round(bboxes[i, 3])))

    # save xml to file
    # print(prettify(annot))
    if not os.path.exists(os.path.join(image_folder, rel_anno_folder)):
        os.mkdir(os.path.join(image_folder, rel_anno_folder))
    output_xml_path = os.path.join(image_folder, rel_anno_folder, '{}.xml'.format(image_name.rsplit('.', 1)[0]))
    print('Writing to {}... '.format(output_xml_path))
    write_xml(output_xml_path, annot)


def write_xml(output_xml_path, tree):
    f = open(output_xml_path, 'w')
    if not isinstance(tree, ET.ElementTree):
        f.write(prettify(tree, '\t'))
    else:
        tree.write(output_xml_path)


def prettify(elem, indent="  "):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    doc = minidom.Document()
    xml_doc = doc.toxml()
    return reparsed.toprettyxml(indent=indent)[(len(xml_doc) + 1):]


if __name__ == '__main__':
    args = get_arguments()
    create_output_files_from_detection(args.image_path, args.model_path, args.label_map_path, args.score_thres,
                                       args.accepted_classes, not args.txt_file)
