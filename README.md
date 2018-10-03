
# Metrics for object detection v2

This a fork of the original **Metrics for object detection** developped by [Rafael Padilla](https://github.com/rafaelpadilla) in [here](https://github.com/rafaelpadilla/Object-Detection-Metrics) so that explains v2. 

## Table of contents

- [Motivation](#metrics-for-object-detection-v2)
- [What this project has to offer?](#what-this-project-has-to-offer)
    - [Other changes in the original project](#other-changes-in-the-original-project)
- [How to use this project](#how-to-use-this-project)
    - [Text files](#text-files)
    - [Xml files](#xml-files)
- [References](#references)
 
## What this project has to offer?

This work was really helpful and clear but somehow lacked some features I wanted so I decided to expand it with some needed features. More precisely this repository can:
* Do exactly what the original repository does (at least at the moment of the forking - 26 September 2018)
* Provide annotation in 2 formats rather in text format only:
    * Text format (txt)
    * Xml format (xml)
* It can use only members of specific classes instead of using all available classes.
* The input format is derived by the extension of files in the provided folders.
* My intension is to provide also a 3<sup>rd</sup> option to provide bounding boxes by using a trained object detection model on some random images.
    
The ability to read xml derives from the way tensorflow annotated the images in object detection module. So, it seems natural to use xml files which are already annotated in a natural manner.

### Other changes in the original project

The code has been modified up to a degree to follow some python conventions. Examples include:
* The use of short options with only 1 sinlge character.
* Organize code with more functions.
* Separate functions to handle different input format.

The functionality has not changed apart from an extra option added (*- -accepted-classes*) to provide the option of applying metric to members of specific classes and of course the conversion of other options to more pythonic accepted standards.

## How to use this project

This project can be used to evaluate the object detection results relatively easy. In order to evaluate the results you need:

* Either txt files for Ground Truth and Detection
* Or xml files for Ground Truth and Detection
* Files for Ground Truth and Detection can be of different format.

### Text files

(This part is the same as the [original code](https://github.com/rafaelpadilla/Object-Detection-Metrics#create-the-ground-truth-files))

These are space delimited text files which contain bounding boxes (bboxes) in either of the two formats in each line, 
either:

`<class_name> <left> <top> <right> <bottom>`

or

`<class_name> <left> <top> <width> <height>`.

The name of the file should be the same between Ground Truth and Detection and the extension is obligatory to be `.txt` (otherwise the code won't be able to determine the input format).

The only difference between Ground Truth and Detection files is that there is second extra value in each line which represent the confidence (or score) of each bbox. So, the actual format becomes:
either

`<class_name> <confidence> <left> <top> <right> <bottom>`

or

`<class_name> <left> <top> <width> <height>`

Default option is `<left> <top> <width> <height>`.

### Xml files

This basically follows the conventions of Pascal Voc annotation scheme. These xml files can be produces by using for example [labelImg](https://github.com/tzutalin/labelImg) on some images and creating manually the annotation.

An xml has roughly this form:
```
<annotation>
	<folder>all</folder>
	<filename>000.jpg</filename>
	<path>groundtruths-xml/000.jpg</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>400</width>
		<height>286</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>person</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>274</xmin>
			<ymin>66</ymin>
			<xmax>351</xmax>
			<ymax>266</ymax>
		</bndbox>
	</object>
	<object>
		<name>person</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>221</xmin>
			<ymin>62</ymin>
			<xmax>269</xmax>
			<ymax>224</ymax>
		</bndbox>
	</object>
</annotation>
```

The main parts of the xml file that are used in this process are:

* `filename` which correspond to the name of the image
* `object/name` which correspond to each bbox class
* `object/bndbox/xmin` which corresponds to the bbox left-most coordinate
* `object/bndbox/ymin` which corresponds to the bbox top-most coordinate
* `object/bndbox/xmax` which corresponds to the bbox right-most coordinate
* `object/bndbox/ymax` which corresponds to the bbox bottom-most coordinate

and for the Detection xml files (these has to be created somehow though):
* `object/score` which corresponds to the confidence of the detected bbox.

### Optional arguments:

| Argument &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Description | Example | Default |
|:-------------:|:-----------:|:-----------:|:-----------:|
| `-h`,<br>`--help ` |	show help message | `python pascalvoc.py -h` | |
|  `-v`,<br>`--version` | check version | `python pascalvoc.py -v` | |
| `-g`,<br>`--gt-folder` | folder that contains the ground truth bounding boxes files | `python pascalvoc.py -g /home/whatever/my_groundtruths/` | `/Object-Detection-Metrics/groundtruths`|
| `-d`,<br>`--det-folder` | folder that contains your detected bounding boxes files | `python pascalvoc.py -d /home/whatever/my_detections/` | `/Object-Detection-Metrics/detections/`|
| `-t`,<br>`--threshold` | IOU thershold that tells if a detection is TP or FP | `python pascalvoc.py -t 0.75` | `0.50` |
| `-gt-format` | format of the coordinates of the ground truth bounding boxes [**\***](#asterisk) | `python pascalvoc.py -gt-format xyrb` | `xywh` |
| `-det-format` | format of the coordinates of the detected bounding boxes [**\***](#asterisk) | `python pascalvoc.py -det-format xyrb` | `xywh` | |
| `-gt-coords` | reference of the ground truth bounding bounding box coordinates.<br>If the annotated coordinates are relative to the image size (as used in YOLO), set it to `rel`.<br>If the coordinates are absolute values, not depending to the image size, set it to `abs` |  `python pascalvoc.py -gt-coords rel` | `abs` |
| `-det-coords` | reference of the detected bounding bounding box coordinates.<br>If the coordinates are relative to the image size (as used in YOLO), set it to `rel`.<br>If the coordinates are absolute values, not depending to the image size, set it to `abs` | `python pascalvoc.py -det-coords rel` | `abs` |
| `-img-size ` | image size in the format `width,height` <int,int>.<br>Required if `-gtcoords` or `-detcoords` is set to `rel` | `python pascalvoc.py -img-size 600,400` |
| `-s`,<br>`--savepath` | folder where the plots are saved | `python pascalvoc.py -s /home/whatever/my_results/` | `Object-Detection-Metrics/results/` |
| `-n`,<br>`--noplot` | if present no plot is shown during execution | `python pascalvoc.py -n` | not presented.<br>Therefore, plots are shown |
| `--accepted-classes` | if present only members belonging to those classes are evaluated (other members are ignored) | `python pascalvoc.py --accepted-classes person car` | empty list.<br>Meaning all classes are taken into consideration |

<a name="asterisk"> </a>
(**\***) set `-gtformat=xywh` and/or `-detformat=xywh` if format is `<left> <top> <width> <height>`. Set to `-gtformat=xyrb` and/or `-detformat=xyrb`  if format is `<left> <top> <right> <bottom>`.
