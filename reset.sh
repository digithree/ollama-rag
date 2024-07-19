#!/bin/bash
read -p "Delete existing configuration and conversation memory. Does not delete PDF DB. Are you sure? (y/n): "

if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm config.json
    echo "Deleted configuration"
    rm Modelfile-generated
    echo "Deleted generated Ollama modelfile"
    rm -rf chroma_db_mem
    echo "Deleted conversation memory"
else
    echo "Aborted"
fi