# this file is for testing only to ensure new shapes and features
from preprossecing.shape_padding import downscale_and_adjust
from PIL import Image


img_path = "C:\datasets\DIV2K\Dataset\DIV2K_train_HR_PAD/0001.png"
image = Image.open(img_path).convert("RGB")
res = downscale_and_adjust()
print(res.shape)
res.show()
