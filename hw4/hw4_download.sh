mkdir -p ckpt/
mkdir -p ckpt/dvgo_hotdog2/

wget https://www.dropbox.com/s/vs47yd1qj0gyr6f/best.pt?dl=1 -O ckpt/best.pt
wget https://www.dropbox.com/s/ce2hu5nokyqc70k/args.txt?dl=1 -O ckpt/dvgo_hotdog2/args.txt
wget https://www.dropbox.com/s/h5wy3zbh0gom9d5/coarse_last.tar?dl=1 -O ckpt/dvgo_hotdog2/coarse_last.tar
wget https://www.dropbox.com/s/gb39zdmit7jags5/config.py?dl=1 -O ckpt/dvgo_hotdog2/config.py
wget https://www.dropbox.com/s/nn8w5txxxh31dt7/fine_last.tar?dl=1 -O ckpt/dvgo_hotdog2/fine_last.tar