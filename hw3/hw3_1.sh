#!/bin/bash
python3 ./zeroshot/infer.py --data_path $1 --json_path $2 --pred_file $3
# TODO - run your inference Python3 code