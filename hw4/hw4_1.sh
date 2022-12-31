#!/bin/bash
python3 ./nvs/run.py --config ./nvs/configs/nerf/hotdog.py --render_only  --infer --json_path $1 --save_path $2 --dump_image