from data_pipe import load_data
import matplotlib.pyplot as pl
import numpy as np

data,filenames, angles = load_data('betta_lap6_data')

img = None
for im,f in list(zip(data,filenames))[::-1]:
    f = f.split('/')[-1]
    im = im.astype(np.uint8)
    if img is None:
        img = pl.imshow(im)
    else:
        img.set_data(im)
    pl.title(f)
    pl.pause(.01)
    pl.draw()