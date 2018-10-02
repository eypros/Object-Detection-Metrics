
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
