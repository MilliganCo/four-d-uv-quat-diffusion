#!/usr/bin/env bash
RUN_DIR=${1:-runs/cifar10}
streamlit run ui/app.py -- --run_dir "$RUN_DIR"

