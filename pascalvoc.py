# coding=utf-8
###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012)                  #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012)                  #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)
# Modified by: George Orfanidis (g.orfanidis@iti.gr)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: 1st Oct 2018                                                 #
###########################################################################################
# v3 separates completely text with xml files

# import os
# print(os.path.realpath(__file__))
import os
import argparse
# from argparse import RawTextHelpFormatter
import glob
import shutil
import sys
import xml.etree.ElementTree as ET

for i, p in enumerate(sys.path):
    if p == '/home/gorfanidis/models':
        del sys.path[i]

# import _init_paths

currentPath = os.path.dirname(os.path.realpath(__file__))

# Add lib to PYTHONPATH
libPath = os.path.join(currentPath, 'lib')
if libPath not in sys.path:
    sys.path.insert(0, libPath)

# from BoundingBox import BoundingBox
# from BoundingBoxes import BoundingBoxes
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
        epilog="Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)")
    # formatter_class=RawTextHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION)
    # Positional arguments
    # Mandatory
    parser.add_argument(
        '-g',
        '--gt-folder',
        dest='gtFolder',
        metavar='',
        help='folder containing your ground truth bounding boxes')
    parser.add_argument(
        '-d',
        '--det-folder',
        dest='detFolder',
        metavar='',
        help='folder containing your detected bounding boxes')
    # Optional
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
        '--det-format',
        dest='detFormat',
        metavar='',
        help='format of the coordinates of the detected bounding boxes '
             '(\'xywh\': <left> <top> <width> <height>) '
             'or (\'xyrb\': <left> <top> <right> <bottom>)')
    parser.add_argument(
        '--gt-coords',
        dest='gtCoordinates',
        metavar='',
        help='reference of the ground truth bounding box coordinates: absolute '
             'values (\'abs\') or relative to its image size (\'rel\')')
    parser.add_argument(
        '--det-coords',
        dest='detCoordinates',
        metavar='',
        help='reference of the ground truth bounding box coordinates: '
             'absolute values (\'abs\') or relative to its image size (\'rel\')')
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
        '--accepted-classes',
        nargs='+',
        default=[],
        help='The list of accepted classes over which to evaluate the model. Ignore all objects ' \
             'belonging to other classes')

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
        return BBFormat.XYWH  # default when nothing is passed
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
                       accepted_classes=[],
                       imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if all_bounding_boxes is None:
        all_bounding_boxes = BoundingBoxes()
    if all_classes is None:
        all_classes = []
    # Read ground truths
    os.chdir(directory)
    files = glob.glob("*.txt")
    if not files:
        files = glob.glob("*.xml")
        files.sort()
        # this assumes files are provided in xml format
        for f in files:
            all_bounding_boxes, all_classes = read_xml_file(os.path.join(directory, f), is_gt, coordType, bbFormat,
                                                            all_bounding_boxes=all_bounding_boxes,
                                                            all_classes=all_classes, accepted_classes=accepted_classes)

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
                                                            accepted_classes=accepted_classes, imgSize=imgSize)
    return all_bounding_boxes, all_classes


def read_txt_file(file_path, is_gt, coordType, bbFormat, all_bounding_boxes=None, all_classes=None,
                  accepted_classes=[], imgSize=(0, 0)):
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
            all_classes.append(id_class)
    fh1.close()
    return all_bounding_boxes, all_classes


def read_xml_file(file_path, is_gt, coordType, bbFormat, all_bounding_boxes=None, all_classes=None,
                  accepted_classes=[]):
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
    # if file_path.endswith('jpg'):
    #     file_path = os.path.join(os.path.dirname(file_path), xml_folder, '{}.xml'
    #                              .format(os.path.basename(file_path).rsplit('.', 1)[0]))
    # if all_bounding_boxes is None:
    #     all_bounding_boxes = BoundingBoxes()
    # if all_classes is None:
    #     all_classes = []
    tree = ET.parse(file_path)
    filename = tree.find('filename').text.rsplit('.', 1)[0]
    img_size = tree.find('size')
    objects = tree.findall('object')
    # bboxes = np.ndarray(shape=(len(objects), 4), dtype=int)
    # labels = []
    for i, obj in enumerate(objects):
        if accepted_classes and obj.find('name').text not in accepted_classes:
            continue
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
            all_classes.append(obj.find('name').text)

        # if width_first:
        #     bboxes[i, :] = [float(obj.find('bndbox').find('xmin').text), float(obj.find('bndbox').find('ymin').text),
        #                     float(obj.find('bndbox').find('xmax').text), float(obj.find('bndbox').find('ymax').text)]
        # else:
        #     bboxes[i, :] = [float(obj.find('bndbox').find('ymin').text), float(obj.find('bndbox').find('xmin').text),
        #                     float(obj.find('bndbox').find('ymax').text), float(obj.find('bndbox').find('xmax').text)]
    return all_bounding_boxes, all_classes  # bboxes, labels


VERSION = '0.3 (beta)'


def validate_args(args):
    """

    :param args:
    """
    iouThreshold = args.iouThreshold

    # Arguments validation
    errors = []
    # Validate formats
    args.gtFormat = validate_formats(args.gtFormat, '-gtformat', errors)
    args.detFormat = validate_formats(args.detFormat, '-detformat', errors)
    # Validate mandatory (paths)
    current_path = os.path.dirname(os.path.abspath(__file__))
    # Groundtruth folder
    if validate_mandatory_args(args.gtFolder, '-gt/--gtfolder', errors):
        args.gtFolder = ValidatePaths(args.gtFolder, '-gt/--gtfolder', errors)
    else:
        errors.pop()
        args.gtFolder = os.path.join(current_path, 'groundtruths')
        if os.path.isdir(args.gtFolder) is False:
            errors.append('folder %s not found' % args.gtFolder)
    # Coordinates types
    args.gtCoordType = validate_coordinates_types(args.gtCoordinates, '-gtCoordinates', errors)
    args.detCoordType = validate_coordinates_types(args.detCoordinates, '-detCoordinates', errors)
    imgSize = (0, 0)
    if args.gtCoordType == CoordinatesType.Relative:  # Image size is required
        args.imgSize = validate_image_size(args.imgSize, '-imgsize', '-gtCoordinates', errors)
    if args.detCoordType == CoordinatesType.Relative:  # Image size is required
        args.imgSize = validate_image_size(args.imgSize, '-imgsize', '-detCoordinates', errors)
    # Detection folder
    if validate_mandatory_args(args.detFolder, '-det/--detfolder', errors):
        args.detFolder = ValidatePaths(args.detFolder, '-det/--detfolder', errors)
    else:
        errors.pop()
        args.detFolder = os.path.join(current_path, 'detections')
        if os.path.isdir(args.detFolder) is False:
            errors.append('folder %s not found' % args.detFolder)
    # Validate savePath
    if args.savePath is not None:
        args.savePath = ValidatePaths(args.savePath, '-sp/--savepath', errors)
        args.savePath = os.path.join(args.savePath, 'results')
    else:
        args.savePath = os.path.join(current_path, 'results')
    # If error, show error messages
    if len(errors) is not 0:
        print("""usage: Object Detection Metrics [-h] [-v] [-gt] [-det] [-t] [-gtformat]
                                    [-detformat] [-save]""")
        print('Object Detection Metrics: error(s): ')
        [print(e) for e in errors]
        sys.exit()

    return args
    # return gtFolder, current_path, gtFormat, detFormat, gtCoordType


def main():
    """
    The main function
    """
    # print('iou_threshold= %f' % iou_threshold)
    # print('savePath = %s' % savePath)
    # print('gt_format = %s' % gt_format)
    # print('det_format = %s' % det_format)
    # print('gt_folder = %s' % gt_folder)
    # print('det_folder = %s' % det_folder)
    # print('gt_coord_type = %s' % gt_coord_type)
    # print('det_coord_type = %s' % det_coord_type)
    # print('show_plot %s' % show_plot)

    args = get_arguments()
    args = validate_args(args)

    gt_folder = args.gtFolder
    det_folder = args.detFolder
    # current_path = args.currentPath
    det_format = args.detFormat
    gt_format = args.gtFormat
    gt_coord_type = args.gtCoordinates
    det_coord_type = args.detCoordinates
    img_size = args.imgSize
    iou_threshold = args.iouThreshold
    save_path = args.savePath
    accepted_classes = args.accepted_classes

    # Create directory to save results
    shutil.rmtree(save_path, ignore_errors=True)  # Clear folder
    os.makedirs(save_path)
    # Show plot during execution
    show_plot = args.showPlot

    all_bounding_boxes, all_classes = get_bounding_boxes(
        gt_folder, True, gt_format, gt_coord_type, accepted_classes=accepted_classes, imgSize=img_size)
    all_bounding_boxes, all_classes = get_bounding_boxes(
        det_folder, False, det_format, det_coord_type, all_bounding_boxes, all_classes,
        accepted_classes=accepted_classes, imgSize=img_size)
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
        # total_TP = metrics_per_class['total TP']
        # total_FP = metrics_per_class['total FP']

        if total_positives > 0:
            valid_classes += 1
            acc_AP += ap
            prec = ['%.2f' % p for p in precision]
            rec = ['%.2f' % r for r in recall]
            ap_str = "{0:.5f}".format(ap)
            # ap_str = str('%.2f' % ap) #AQUI
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


if __name__ == '__main__':
    main()
