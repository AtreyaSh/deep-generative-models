#!/bin/bash
set -e
wget http://conradsanderson.id.au/lfwcrop/lfwcrop_grey.zip
unzip lfwcrop_grey.zip -d ./data/
mv lfwcrop_grey.zip ./data
cd ./data
ln -s lfwcrop_grey/faces .
