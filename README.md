
# Metrics for object detection v2

This a fork of the original **Metrics for object detection** developped by [Rafael Padilla](https://github.com/rafaelpadilla) in [here](https://github.com/rafaelpadilla/Object-Detection-Metrics) so that explains v2. 

## Table of contents

- [Motivation](#metrics-for-object-detection-v2)
- [What this project has to offer?](#what-this-project-has-to-offer)
- [How to use this project](#how-to-use-this-project)
- [Evaluation using pascalvoc](#evaluation-using-pascalvoc)
    - [Examples of use](#examples-of-use)
    - [Text files](#text-files)
    - [Xml files](#xml-files)
    - [Optional Optional arguments for pascalvoc](#optional-arguments-for-pascalvoc)
- [Applying an object detector to an image folder](#applying-an-object-detector-to-an-image-folder)
    - [Examples of use](#examples-of-use-1)
    - [Optional arguments for detect_bboxes](#optional-arguments-for-detect_bboxes)
- [Direct evaluation of an object detector over an image folder](#direct-evaluation-of-an-object-detector-over-an-image-folder)
    - [Optional arguments for eval_model](#Optional-arguments-for-eval_model)
    - [Examples of use](#examples-of-use-2)

## What this project has to offer?

This work was really helpful and clear but somehow lacked some features I wanted so I decided to expand it with some needed features. More precisely this repository can:
* Do exactly what the original repository does (at least at the moment of the forking - 26 September 2018)
* Provide annotation in 2 formats rather in text format only:
    * Text format (txt)
    * Xml format (xml)
* It can use only members of specific classes instead of using all available classes.
* The input format is derived by the extension of files in the provided folders.
* The project offers the options of using a model to detect object in image folder and produce bbox files (xml or txt).
* (Fixed) The project provides also a 3<sup>rd</sup> option to get the detection bounding boxes by using a trained object detection model on some random images folder.

All 3 cases of course require the existence of ground truth files (txt or xml).

The ability to read xml derives from the way tensorflow annotated the images in object detection module. So, it seems natural to use xml files which are already annotated in a natural manner.

## How to use this project

This project can be used to evaluate the object detection results relatively easy. Currently, there are 3 (overlapping) ways to evaluate a model.

1. Use text (or xml) files containing the bboxes of both ground truth and detection cases. In this case the proposed method is [`pascalvoc`](#evaluation-using-pascalvoc).

1. Use text (or xml) files for the ground truth bboxes and use an object detection model to create (predict) bboxes over an image folder. In this case the proposed method would be [`detect_bboxes`](#applying-an-object-detector-to-an-image-folder) followed by [`pascalvoc`](#evaluation-using-pascalvoc).
In this case `detect_bboxes` is used to create the bboxes in a folder and then the process is identical to the previous case.

1. The final option includes the use of txt (or xml) files for bboxes and application of [`eval_model`](#direct-evaluation-of-an-object-detector-over-an-image-folder).
This option is the more discrete leaving no intermediate outputs.


## Evaluation using pascalvoc

This case uses already present bboxes in text or xml files.

In order to evaluate the results you need:

* Either txt files for Ground Truth and Detection
* Or xml files for Ground Truth and Detection
* Files for Ground Truth and Detection can be of different format.


### Examples of use

The simplest example for the given folders would be:

`python3 pascalvoc.py --accepted-classes person`

which will evaluate the xml files in *detections-xml* over the xml files in *groundtruths-xml* subfolder.

The above assumes two sub-folders *groundtruths-xml* and *detections-xml* containing xml files exist in the same folder of the project.

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

### Optional arguments for pascalvoc:

| Argument &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Description | Example | Default |
|:-------------:|:-----------:|:-----------:|:-----------:|
| `-h`,<br>`--help ` |	show help message | `python pascalvoc.py -h` | |
|  `-v`,<br>`--version` | check version | `python pascalvoc.py -v` | |
| `-g`,<br>`--gt-folder` | folder that contains the ground truth bounding boxes files | `python pascalvoc.py -g /home/whatever/my_groundtruths/` | `/Object-Detection-Metrics/groundtruths-xml/`|
| `-d`,<br>`--det-folder` | folder that contains your detected bounding boxes files | `python pascalvoc.py -d /home/whatever/my_detections/` | `/Object-Detection-Metrics/detections-xml/`|
| `-t`,<br>`--threshold` | IOU thershold that tells if a detection is TP or FP | `python pascalvoc.py -t 0.75` | `0.50` |
| `--gt-format` | format of the coordinates of the ground truth bounding boxes [**\***](#asterisk) | `python pascalvoc.py --gt-format xyrb` | `xyrb` |
| `--det-format` | format of the coordinates of the detected bounding boxes [**\***](#asterisk) | `python pascalvoc.py --det-format xyrb` | `xyrb` | |
| `--gt-coords` | reference of the ground truth bounding bounding box coordinates.<br>If the annotated coordinates are relative to the image size (as used in YOLO), set it to `rel`.<br>If the coordinates are absolute values, not depending to the image size, set it to `abs` |  `python pascalvoc.py --gt-coords rel` | `abs` |
| `--det-coords` | reference of the detected bounding bounding box coordinates.<br>If the coordinates are relative to the image size (as used in YOLO), set it to `rel`.<br>If the coordinates are absolute values, not depending to the image size, set it to `abs` | `python pascalvoc.py --det-coords rel` | `abs` |
| `--img-size ` | image size in the format `width,height` <int,int>.<br>Required if `--gt-coords` or `--det-coords` is set to `rel` | `python pascalvoc.py --img-size 600,400` |
| `-s`,<br>`--savepath` | folder where the plots are saved | `python pascalvoc.py -s /home/whatever/my_results/` | `Object-Detection-Metrics/results/` |
| `-n`,<br>`--noplot` | if present no plot is shown during execution | `python pascalvoc.py -n` | not presented.<br>Therefore, plots are shown |
| `--accepted-classes` | if present only members belonging to those classes are evaluated (other members are ignored) | `python pascalvoc.py --accepted-classes person car` | empty list.<br>Meaning all classes are taken into consideration |

<a name="asterisk"> </a>
(**\***) set `-gtformat=xywh` and/or `-detformat=xywh` if format is `<left> <top> <width> <height>`. Set to `-gtformat=xyrb` and/or `-detformat=xyrb`  if format is `<left> <top> <right> <bottom>`.

## Applying an object detector to an image folder

The project can use an (already trained) object detector to predict bboxes on an image folder. In order to apply this feature you need:

- A trained object detection model (the frozen one to be morer specific)
- A label map which maps objects ids with their respective (human readable) labels


The steps required are roughly:

- The project can use a (tensorflow) object detection model already trained to produce xml or txt files using:
`detect_bboxes.py`. Currently only tensorflow object detector are supported.
- The output can be either txt or xml files.
- After the bboxes have been saved to folder `pascalvoc.py` can be applied to evaluate the performance of the model.


### Examples of use

`--accepted-classes person` is necessary because the examples given contains ground truth  samples of multiple classes but detection was performed on class person only.
If run without this argument the per class AP would be correct for `person` but `0.0` for other classes and `mAP` would have taken into consideration all classes.

To run `detect_bboxes` the simpler example would be:

`python3 create_detection_bboxes.py -i image-samples/ -m /path/to/model -l path/to/label_map.pbtxt`

which will produce xml files to a subfolder of the image folder.

### Optional arguments for detect_bboxes:

| Argument &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Description | Example | Default |
|:-------------:|:-----------:|:-----------:|:-----------:|
| `-h`,<br>`--help ` |	show help message | `python detect_bboxes.py -h` | |
|  `-v`,<br>`--version` | check version | `python detect_bboxes.py -v` | |
| `-i`,<br>`--image_path` | folder containing the images to apply the detection model | `python detect_bboxes.py -i /home/whatever/my_images/` | |
| `-m`,<br>`--model-path` | folder containing the trained model path. This is the folder where the "frozen_inference_graph.pb" resides in other words. | `python detect_bboxes.py -m /home/whatever/my_model/` | |
| `-l`,<br>`--label-map-path` | the path to the label map for this model. | `python detect_bboxes.py -l /path/to/label_map` | |
| `--score-thres` | the threshold under which the bboxes will be ignored and not written to the output files. Default value is 0.0 | `python detect_bboxes.py --score-thres=0.2` | `0.0` |
| `--accepted-classes` | A list with all classes to be taken into consideration when writing the bboxes in files. Default value is an empty list which corresponds to take into consideration all available classes | `python detect_bboxes.py --accepted-classes person car` | empty list, which means all samples are treated | |
| `-t`,<br>`--txt_file` | Whether output file will be xml or txt. If not set xml is used (default). |  `python detect_bboxes.py --txt_file` | xml |


## Direct evaluation of an object detector over an image folder

The 3<sup>rd</sup> option uses an object detector over an image folder for the evaluation of the same model performance.
The difference with the 2<sup>nd</sup> option is that in this scenario no intermediate files are created and there is just the output of the evaluation.
Also this mode takes the more arguments since it combines elements from both previous cases.

As regards performance, the 3 methods use essentially the same tools so besides the automation and file writing overhead gained in the 3<sup>rd</sup> case no other differences exists.

### Examples of use

The simpler use would be:

`python3 eval_model.py -g path/to/gt -i path/to/image_folder -m path/to/model -l path/to/label_map`

while a potentially more versatile use (applied for only classes `person` and `car`) and just print the mAP (without plotting anything):

`python3 eval_model.py -g path/to/gt -i path/to/image_folder -m path/to/model -l path/to/label_map -n --accepted-classes person car`

### Optional arguments for eval_model:

| Argument &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Description | Example | Default |
|:-------------:|:-----------:|:-----------:|:-----------:|
| `-h`,<br>`--help ` |	show help message | `python eval_model.py -h` | |
| `-v`,<br>`--version` | check version | `python eval_model.py -v` | |
| `-g`,<br>`--gt-folder` | folder that contains the ground truth bounding boxes files | `python eval_model.py -g /home/whatever/my_groundtruths/` | `/Object-Detection-Metrics/groundtruths-xml/`|
| `-t`,<br>`--threshold` | IOU thershold that tells if a detection is TP or FP | `python eval_model.py -t 0.75` | `0.50` |
| `--gt-format` | format of the coordinates of the ground truth bounding boxes | `python eval_model.py --gt-format xyrb` | `xyrb` |
| `--gt-coords` | reference of the ground truth bounding bounding box coordinates.<br>If the annotated coordinates are relative to the image size (as used in YOLO), set it to `rel`.<br>If the coordinates are absolute values, not depending to the image size, set it to `abs` |  `python eval_model.py --gt-coords rel` | `abs` |
| `--img-size ` | image size in the format `width,height` <int,int>.<br>Required if `--gt-coords` or `--det-coords` is set to `rel` | `python eval_model.py --img-size 600,400` |
| `-s`,<br>`--savepath` | folder where the plots are saved | `python eval_model.py -s /home/whatever/my_results/` | `Object-Detection-Metrics/results/` |
| `-n`,<br>`--noplot` | if present no plot is shown during execution | `python eval_model.py -n` | not presented.<br>Therefore, plots are shown |
| `-m`,<br>`--model-path` | folder containing the trained model path. This is the folder where the "frozen_inference_graph.pb" resides in other words. | `python eval_model.py -m /home/whatever/my_model/` | |
| `-l`,<br>`--label-map-path` | the path to the label map for this model. | `python eval_model.py -l /path/to/label_map` | |
| `--score-thres` | the threshold under which the bboxes will be ignored and not written to the output files. Default value is 0.0 | `python eval_model.py --score-thres=0.2` | `0.0` |
| `--accepted-classes` | A list with all classes to be taken into consideration when writing the bboxes in files. Default value is an empty list which corresponds to take into consideration all available classes | `python detect_bboxes.py --accepted-classes person car` | empty list, which means all samples are treated | |
| `--merged-classes` | A path to a json file containing a dict for merging classes in ground truth bounding boxes\n' | `python detect_bboxes.py --merged-classes path/to/merged_class.json` | empty dict, which means no merging occurs | |
