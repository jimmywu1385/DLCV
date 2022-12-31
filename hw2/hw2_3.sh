#!/bin/bash
if [[ $1 == *"svhn"* ]]
then
python3 ./uda/test.py --target svhn --model_name svhn.pt --infer_path $1 --pred_file $2
else
python3 ./uda/test.py --target usps --model_name usps.pt --infer_path $1 --pred_file $2
fi
# TODO - run your inference Python3 code