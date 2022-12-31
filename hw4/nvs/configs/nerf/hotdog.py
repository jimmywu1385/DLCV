_base_ = '../default.py'

expname = 'dvgo_hotdog2'
basedir = './ckpt'

data = dict(
    datadir='./hw4_data/hotdog',
    dataset_type='blender',
    white_bkgd=True,
)

