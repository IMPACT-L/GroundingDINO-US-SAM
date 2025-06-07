#!/bin/bash

root="/home/hamze/Documents/Grounding-Sam-Ultrasound/"

rm "$root/multimodal-data/test.CSV" 
rm "$root/multimodal-data/train.CSV" 
rm "$root/multimodal-data/val.CSV" 

rm -r "$root/multimodal-data/test_image/" 
rm -r "$root/multimodal-data/test_mask/" 
rm -r "$root/multimodal-data/train/" 
rm -r "$root/multimodal-data/val/" 

