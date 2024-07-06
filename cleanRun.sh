#!/bin/bash
read -p "Delete previous memory and launch. Are you sure? (y/n): "

if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf chroma_db_mem
    echo "Removed memory. Launching..."
    streamlit run app.py
else
    echo "Aborted"
fi