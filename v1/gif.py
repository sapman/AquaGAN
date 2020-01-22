import glob
from PIL import Image

# filepaths
fp_in = "./images/*"
fp_out = "./image.gif"
# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f).resize(size=(400, 175)) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='gif', append_images=imgs,
         save_all=True, duration=200, loop=0)