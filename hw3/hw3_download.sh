#!/bin/bash
python3 -c "import clip; clip.load('ViT-B/32')"
mkdir -p ckpt/

wget https://www.dropbox.com/s/asbwubzfq2kuc07/caption_tokenizer.json?dl=1 -O ckpt/caption_tokenizer.json
wget https://www.dropbox.com/s/1hfzsfywwwqduu2/caption.pt?dl=1 -O ckpt/caption.pt