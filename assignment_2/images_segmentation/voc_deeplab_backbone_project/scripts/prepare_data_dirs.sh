#!/usr/bin/env bash
set -e
mkdir -p data/VOCdevkit/VOC2012/JPEGImages
mkdir -p data/VOCdevkit/VOC2012/SegmentationClass
mkdir -p data/VOCdevkit/VOC2012/ImageSets/Segmentation
printf "Created expected VOC directory structure under data/VOCdevkit/VOC2012\n"
