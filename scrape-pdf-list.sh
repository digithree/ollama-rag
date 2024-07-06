#!/bin/bash

searchpath=$1

# Check if there is not exactly 1 parameter
if [ $# -ne 1 ]; then
  echo "Error: Expected exactly 1 parameter, the path to root folder of PDF files"
  exit 1
fi

# If no options are passed, print usage text
if [ -z "${searchpath}" ]; then
    echo "Error: Expected exactly 1 parameter, the path to root folder of PDF files"
    usage
fi

find "$1" -type f -exec readlink -f {} \; | grep -i .pdf >> pdf-files.txt