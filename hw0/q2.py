from PIL import Image as im
im1 = im.open("Lena.png")
im1.rotate(180).save("ans2.png")