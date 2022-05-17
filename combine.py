import sys
from PIL import Image

# courtesy of this overflow post
# https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
arrs = [
  ('all.png', ['ada.png', 'div.png']),
  ('adx.png', ['rho.png', 'err.png'])
]

for fname, arr in arrs:
  images = [Image.open(x) for x in arr]
  widths, heights = zip(*(i.size for i in images))

  total_height = sum(heights)
  max_width = max(widths)

  new_im = Image.new('RGB', (max_width, total_height))

  y_offset = 0
  for im in images:
    new_im.paste(im, (0, y_offset))
    y_offset += im.size[1]

  new_im.save(fname)