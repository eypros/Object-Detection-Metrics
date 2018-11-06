# coding=utf-8
###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detection model by  applying the following     #
# metrics:                                                                                #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012)                  #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012)                  #
#                                                                                         #
# Developed by: George Orfanidis (g.orfanidis@iti.gr)                                     #
#        Last modification: 6th Nov 2018                                                  #
###########################################################################################
import json
import os
import argparse
# from argparse import RawTextHelpFormatter
from glob import glob
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
import tensorflow as tf
from PIL import Image

currentPath = os.path.dirname(os.path.realpath(__file__))

# Add lib to PYTHONPATH
libPath = os.path.join(currentPath, 'lib')
if libPath not in sys.path:
    sys.path.insert(0, libPath)

from Evaluator import *
from utils_eval import BBFormat
from utils_eval import BBType
from utils_eval import CoordinatesType


def get_arguments():
    """
    Parse the arguments
    :return:
    """
    parser = argparse.ArgumentParser(
        prog='Object Detection Metrics - Pascal VOC',
        description='This project applies the most popular metrics used to evaluate object detection '
                    'algorithms.\nThe current implemention runs the Pascal VOC metrics.\nFor further references, '
                    'please check:\nhttps://github.com/rafaelpadilla/Object-Detection-Metrics',
        epilog="Developed by: George Orfanidis (g.orfanidis@iti.gr)")
    # formatter_class=RawTextHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION)
    parser.add_argument(
        '-g',
        '--gt-folder',
        dest='gtFolder',
        metavar='',
        help='folder containing your ground truth bounding boxes')
    parser.add_argument(
        '-t',
        '--threshold',
        dest='iouThreshold',
        type=float,
        default=0.5,
        metavar='',
        help='IOU threshold. Default 0.5')
    parser.add_argument(
        '--gt-format',
        dest='gtFormat',
        metavar='',
        help='format of the coordinates of the ground truth bounding boxes: '
             '(\'xywh\': <left> <top> <width> <height>)'
             ' or (\'xyrb\': <left> <top> <right> <bottom>)')
    parser.add_argument(
        '--gt-coords',
        dest='gtCoordinates',
        metavar='',
        help='reference of the ground truth bounding box coordinates: absolute '
             'values (\'abs\') or relative to its image size (\'rel\')')
    parser.add_argument(
        '--img-size',
        dest='imgSize',
        metavar='',
        help='image size. Required if -gtcoords or -detcoords are \'rel\'')
    parser.add_argument(
        '-s', '--save-path', dest='savePath', metavar='', help='folder where the plots are saved')
    parser.add_argument(
        '-n',
        '--no-plot',
        dest='showPlot',
        action='store_false',
        help='no plot is shown during execution')
    parser.add_argument(
        '-i',
        '--image-path',
        help='folder containing the images to apply the detection model')
    parser.add_argument(
        '-m',
        '--model-path',
        help='folder containing the trained model path. This is the folder where the "frozen_inference_graph.pb" '
             'resides in other words')
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
        nargs='+',
        help='A list with all classes to be taken into consideration when writing the bboxes in files.\n'
             'Default value is an empty list which corresponds to take into consideration all available classes')
    parser.add_argument(
        '--merged-classes',
        metavar='',
        help='A path to a json file containing a dict for merging classes in ground truth bounding boxes\n'
             'If GT bounding boxes contain "bike" and "motorcycle" but model was trained on a merged "moto-bike"\n'
             'a dict {\'bike\': \'moto-bike\', \'motorcycle\': \'moto-bike\' should be used.')

    args = parser.parse_args()

    return args


# Validate formats
def validate_formats(arg_format, arg_name, errors):
    """

    :param arg_format:
    :param arg_name:
    :param errors:
    :return:
    """
    if arg_format == 'xywh':
        return BBFormat.XYWH
    elif arg_format == 'xyrb':
        return BBFormat.XYX2Y2
    elif arg_format is None:
        return BBFormat.XYX2Y2  # default when nothing is passed
    else:
        errors.append(
            'argument %s: invalid value. It must be either \'xywh\' or \'xyrb\'' % arg_name)


# Validate mandatory args
def validate_mandatory_args(arg, arg_name, errors):
    """

    :param arg:
    :param arg_name:
    :param errors:
    :return:
    """
    if arg is None:
        errors.append('argument %s: required argument' % arg_name)
    else:
        return True


def validate_image_size(arg, arg_name, arg_informed, errors):
    """

    :param arg:
    :param arg_name:
    :param arg_informed:
    :param errors:
    :return:
    """
    errorMsg = 'argument %s: required argument if %s is relative' % (arg_name, arg_informed)
    ret = None
    if arg is None:
        errors.append(errorMsg)
    else:
        arg = arg.replace('(', '').replace(')', '')
        args = arg.split(',')
        if len(args) != 2:
            errors.append(
                '%s. It must be in the format \'width,height\' (e.g. \'600,400\')' % errorMsg)
        else:
            if not args[0].isdigit() or not args[1].isdigit():
                errors.append(
                    '%s. It must be in INdiaTEGER the format \'width,height\' (e.g. \'600,400\')' %
                    errorMsg)
            else:
                ret = (int(args[0]), int(args[1]))
    return ret


# Validate coordinate types
def validate_coordinates_types(arg, arg_name, errors):
    """

    :param arg:
    :param arg_name:
    :param errors:
    :return:
    """
    if arg == 'abs':
        return CoordinatesType.Absolute
    elif arg == 'rel':
        return CoordinatesType.Relative
    elif arg is None:
        return CoordinatesType.Absolute  # default when nothing is passed
    errors.append('argument %s: invalid value. It must be either \'rel\' or \'abs\'' % arg_name)


def ValidatePaths(arg, name_arg, errors):
    """

    :param arg:
    :param name_arg:
    :param errors:
    :return:
    """
    if arg is None:
        errors.append('argument %s: invalid directory' % name_arg)
    elif os.path.isdir(arg) is False and os.path.isdir(os.path.join(currentPath, arg)) is False:
        errors.append('argument %s: directory does not exist \'%s\'' % (name_arg, arg))
    elif os.path.isdir(os.path.join(currentPath, arg)) is True:
        arg = os.path.join(currentPath, arg)
    return arg


def get_bounding_boxes(directory,
                       is_gt,
                       bbFormat,
                       coordType,
                       all_bounding_boxes=None,
                       all_classes=None,
                       accepted_classes=None,
                       imgSize=(0, 0),
                       merged_classes=None):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if merged_classes is None:
        merged_classes = {}
    if accepted_classes is None:
        accepted_classes = []
    if all_bounding_boxes is None:
        all_bounding_boxes = BoundingBoxes()
    if all_classes is None:
        all_classes = set()
    # Read ground truths
    os.chdir(directory)
    files = glob("*.txt")
    if not files:
        files = glob("*.xml")
        files.sort()
        # this assumes files are provided in xml format
        for f in files:
            all_bounding_boxes, all_classes = read_xml_file(os.path.join(directory, f), is_gt, coordType, bbFormat,
                                                            all_bounding_boxes=all_bounding_boxes,
                                                            all_classes=all_classes, accepted_classes=accepted_classes,
                                                            merged_classes=merged_classes)

    else:
        files.sort()
        # Read GT detections from txt file
        # Each line of the files in the groundtruths folder represents a ground truth bounding box
        # (bounding boxes that a detector should detect)
        # Each value of each line is  "class_id, x, y, width, height" respectively
        # Class_id represents the class of the bounding box
        # x, y represents the most top-left coordinates of the bounding box
        # x2, y2 represents the most bottom-right coordinates of the bounding box
        for f in files:
            all_bounding_boxes, all_classes = read_txt_file(f, is_gt, coordType, bbFormat,
                                                            all_bounding_boxes=all_bounding_boxes,
                                                            all_classes=all_classes,
                                                            accepted_classes=accepted_classes, imgSize=imgSize,
                                                            merged_classes=merged_classes)
    return all_bounding_boxes, all_classes


def get_bounding_boxes_by_detection(model_path, label_map_path, image_path, score_thres,
                                    all_bounding_boxes=None,
                                    all_classes=None,
                                    accepted_classes=None,
                                    img_size=(0, 0)):
    """
    Apply object detection to images and return the detected bounding boxes
    :param model_path:
    :param label_map_path:
    :param image_path:
    :param score_thres:
    :param all_bounding_boxes:
    :param all_classes:
    :param accepted_classes:
    :param img_size:
    :return:
    """
    if accepted_classes is None:
        accepted_classes = []
    if all_bounding_boxes is None:
        all_bounding_boxes = BoundingBoxes()
    if all_classes is None:
        all_classes = set()
    if not label_map_path:
        label_map_path = os.path.join(model_path, 'pipeline.config') if \
            os.path.exists(os.path.join(model_path, 'pipeline.config')) else os.path.join(os.path.dirname(
            os.path.dirname(os.path.join(model_path))), 'pipeline.config')
    categories_dict = get_label_map(label_map_path, name_to_id=False)  # {1: 'person', 2: 'car'}

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
                image_name = os.path.basename(im).split('.', 1)[0]
                print('Processing {}...'.format(im))
                image = Image.open(im)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes_det, scores_det, labels_det_ids, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                boxes_det, scores_det, labels_det_ids = filter_bboxes_by_score(boxes_det, scores_det, labels_det_ids,
                                                                               image_np, score_thres=score_thres,
                                                                               width_first=True)
                for i in range(boxes_det.shape[0]):
                    if accepted_classes and categories_dict[labels_det_ids[i]] not in accepted_classes:
                        continue
                    bb = BoundingBox(
                        image_name,
                        categories_dict[labels_det_ids[i]],
                        boxes_det[i, 0],
                        boxes_det[i, 1],
                        boxes_det[i, 2],
                        boxes_det[i, 3],
                        CoordinatesType.Absolute,
                        img_size,
                        BBType.Detected,
                        scores_det[i],
                        format=BBFormat.XYX2Y2)
                    all_bounding_boxes.addBoundingBox(bb)
                    all_classes.add(categories_dict[labels_det_ids[i]])
    end_det = time.time() - start_det

    return all_bounding_boxes, all_classes, end_det


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


def load_image_into_numpy_array(image_input):
    """

    :param image_input:
    :return:
    """
    (im_width, im_height) = image_input.size
    return np.array(image_input.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


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


def read_txt_file(file_path, is_gt, coordType, bbFormat, all_bounding_boxes=None, all_classes=None,
                  accepted_classes=None, imgSize=(0, 0)):
    """
    Reads a txt file to get all the information needed (bbounding boxes etc)
    :param file_path:
    :param is_gt:
    :param coordType:
    :param bbFormat:
    :param all_bounding_boxes:
    :param all_classes:
    :param accepted_classes:
    :param imgSize:
    :return:
    """
    if accepted_classes is None:
        accepted_classes = []
    nameOfImage = file_path.replace(".txt", "")
    fh1 = open(file_path, "r")
    for line in fh1:
        line = line.replace("\n", "")
        if line.replace(' ', '') == '':
            continue
        splitLine = line.split(" ")
        if accepted_classes and splitLine[0] not in accepted_classes:
            continue
        if is_gt:
            # id_class = int(splitLine[0]) #class
            id_class = (splitLine[0])  # class
            x = float(splitLine[1])
            y = float(splitLine[2])
            w = float(splitLine[3])
            h = float(splitLine[4])
            bb = BoundingBox(
                nameOfImage,
                id_class,
                x,
                y,
                w,
                h,
                coordType,
                imgSize,
                BBType.GroundTruth,
                format=bbFormat)
        else:
            # id_class = int(splitLine[0]) #class
            id_class = (splitLine[0])  # class
            confidence = float(splitLine[1])
            x = float(splitLine[2])
            y = float(splitLine[3])
            w = float(splitLine[4])
            h = float(splitLine[5])
            bb = BoundingBox(
                nameOfImage,
                id_class,
                x,
                y,
                w,
                h,
                coordType,
                imgSize,
                BBType.Detected,
                confidence,
                format=bbFormat)
        all_bounding_boxes.addBoundingBox(bb)
        if id_class not in all_classes:
            all_classes.add(id_class)
    fh1.close()
    return all_bounding_boxes, all_classes


def read_xml_file(file_path, is_gt, coordType, bbFormat, all_bounding_boxes=None, all_classes=None,
                  accepted_classes=None, merged_classes=list()):
    """
    Reads an xml file
    :param file_path:
    :param is_gt:
    :param coordType:
    :param bbFormat:
    :param all_bounding_boxes:
    :param all_classes:
    :param accepted_classes:
    :return:
    """
    if accepted_classes is None:
        accepted_classes = []
    tree = ET.parse(file_path)
    filename = tree.find('filename').text.rsplit('.', 1)[0]
    img_size = tree.find('size')
    objects = tree.findall('object')
    for i, obj in enumerate(objects):
        # if accepted_classes and obj.find('name').text not in accepted_classes:
        # if accepted_classes and (obj['name'] not in accepted_classes or (True if (obj['name'] in merged_classes and
        #                                                                           merged_classes[obj['name']] not in
        #                                                                           merged_classes) else False)):
        # print(obj.find('name').text)
        if accepted_classes and (obj.find('name').text not in accepted_classes and
                                 (False if (obj.find('name').text in merged_classes and merged_classes[obj.find('name').text] in accepted_classes) else True)):
        # if accepted_classes and (obj.find('name').text not in accepted_classes and (False if (obj.find('name').text in
        #                                                                             merged_classes and
        #                                                                             merged_classes[
        #                                                                                 obj.find('name').text]
        #                                                                             not in accepted_classes) else True)):
                continue
        if obj.find('name').text in merged_classes:
            obj.find('name').text = merged_classes[obj.find('name').text]
        if not is_gt:
            bb = BoundingBox(
                filename,
                obj.find('name').text,
                float(obj.find('bndbox').find('xmin').text),
                float(obj.find('bndbox').find('ymin').text),
                float(obj.find('bndbox').find('xmax').text),
                float(obj.find('bndbox').find('ymax').text),
                coordType,
                img_size,
                BBType.Detected,
                float(obj.find('score').text),
                format=bbFormat)
        else:
            bb = BoundingBox(
                filename,
                obj.find('name').text,
                float(obj.find('bndbox').find('xmin').text),
                float(obj.find('bndbox').find('ymin').text),
                float(obj.find('bndbox').find('xmax').text),
                float(obj.find('bndbox').find('ymax').text),
                coordType,
                img_size,
                BBType.GroundTruth,
                format=bbFormat)
        all_bounding_boxes.addBoundingBox(bb)
        if obj.find('name').text not in all_classes:
            all_classes.add(obj.find('name').text)

    return all_bounding_boxes, all_classes


VERSION = '0.3 (beta)'


def validate_args(args):
    """

    :param args:
    """
    # Arguments validation
    errors = []
    # Validate formats
    args.gtFormat = validate_formats(args.gtFormat, '--gt-format', errors)
    # args.detFormat = validate_formats(args.detFormat, '--det-format', errors)
    # Validate mandatory (paths)
    current_path = os.path.dirname(os.path.abspath(__file__))
    # Groundtruth folder
    if validate_mandatory_args(args.gtFolder, '-gt/--gt-folder', errors):
        args.gtFolder = ValidatePaths(args.gtFolder, '-gt/--gt-folder', errors)
    else:
        errors.pop()
        args.gtFolder = os.path.join(current_path, 'groundtruths-xml')
        if os.path.isdir(args.gtFolder) is False:
            errors.append('folder %s not found' % args.gtFolder)
    # Coordinates types
    args.gtCoordType = validate_coordinates_types(args.gtCoordinates, '--gt-coords', errors)
    # args.detCoordType = validate_coordinates_types(args.detCoordinates, '-det-ccords', errors)
    if args.gtCoordType == CoordinatesType.Relative:  # Image size is required
        args.imgSize = validate_image_size(args.imgSize, '-imgsize', '--gt-coords', errors)

    # Validate savePath
    if args.savePath is not None:
        args.savePath = ValidatePaths(args.savePath, '-s/--save-path', errors)
        args.savePath = os.path.join(args.savePath, 'results')
    else:
        args.savePath = os.path.join(current_path, 'results')
    # If error, show error messages
    if len(errors) is not 0:
        print("""usage: Object Detection Metrics [-h] [-v] [-g] [-d] [-t] [--gt-format]
                                    [--det-format] [-save]""")
        print('Object Detection Metrics: error(s): ')
        [print(e) for e in errors]
        sys.exit()

    return args


def main():
    """
    The main function
    """
    start = time.time()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    args = get_arguments()
    args = validate_args(args)
    print('Starting evaluation of object detection model in {} for images in {}...'.format(args.model_path, args.image_path))

    gt_folder = args.gtFolder
    gt_format = args.gtFormat
    gt_coord_type = args.gtCoordinates
    img_size = args.imgSize
    iou_threshold = args.iouThreshold
    save_path = args.savePath
    accepted_classes = args.accepted_classes
    if args.merged_classes:
        with open(args.merged_classes)  as fp:
            merged_classes = json.load(fp)
    else:
        merged_classes = None

    # Create directory to save results
    shutil.rmtree(save_path, ignore_errors=True)  # Clear folder
    os.makedirs(save_path)
    # Show plot during execution
    show_plot = args.showPlot

    all_bounding_boxes, all_classes = get_bounding_boxes(
        gt_folder, True, gt_format, gt_coord_type, accepted_classes=accepted_classes, imgSize=img_size,
        merged_classes=merged_classes)
    all_bounding_boxes, all_classes, det_time = get_bounding_boxes_by_detection(args.model_path, args.label_map_path,
                                                                                args.image_path, args.score_thres,
                                                                                all_bounding_boxes, all_classes,
                                                                                accepted_classes,
                                                                                img_size)
    all_classes = list(all_classes)
    all_classes.sort()

    f = open(os.path.join(save_path, 'results.txt'), 'w')
    f.write('Object Detection Metrics\n')
    f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
    f.write('Average Precision (AP), Precision and Recall per class:')

    evaluator = Evaluator()
    acc_AP = 0
    valid_classes = 0
    # for each class
    for c in all_classes:
        # Plot Precision x Recall curve
        metrics_per_class = evaluator.plot_precision_recall_curve(
            c,  # Class to show
            all_bounding_boxes,  # Object containing all bounding boxes (ground truths and detections)
            IOUThreshold=iou_threshold,  # IOU threshold
            showAP=True,  # Show Average Precision in the title of the plot
            show_interpolated_precision=False,  # Don't plot the interpolated precision curve
            save_path=os.path.join(save_path, c + '.png'),
            show_graphics=show_plot)
        # Get metric values per each class
        cl = metrics_per_class['class']
        ap = metrics_per_class['AP']
        precision = metrics_per_class['precision']
        recall = metrics_per_class['recall']
        total_positives = metrics_per_class['total positives']

        if total_positives > 0:
            valid_classes += 1
            acc_AP += ap
            prec = ['%.2f' % p for p in precision]
            rec = ['%.2f' % r for r in recall]
            ap_str = "{0:.5f}".format(ap)
            print('AP: %s (%s)' % (ap_str, cl))
            f.write('\n\nClass: %s' % cl)
            f.write('\nAP: %s' % ap_str)
            f.write('\nPrecision: %s' % prec)
            f.write('\nRecall: %s' % rec)

    mAP = acc_AP / valid_classes
    mAP_str = "{0:.5f}".format(mAP)
    print('mAP: %s' % mAP_str)
    f.write('\n\n\nmAP: %s' % mAP_str)
    f.close()
    end = time.time()
    print('Elapsed time in detection was {:2.3f} secs'.format(det_time))
    print('Total elapsed time was {:2.3f} secs'.format(end - start))


if __name__ == '__main__':
    main()
