#!/bin/bash
set -e
wget http://conradsanderson.id.au/lfwcrop/lfwcrop_grey.zip
unzip lfwcrop_grey.zip
ln -s lfwcrop_grey/faces .
