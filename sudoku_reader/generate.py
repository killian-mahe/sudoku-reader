from PIL import Image, ImageDraw, ImageFont
import numpy as np
import ntpath
import os
import glob

fontSize = 26
imgSize = (28, 28)
init_position = (7, -1)

dataset_path = os.path.join(os.getcwd(), "Synthetic_dataset")
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

fhandle = open("Fonts_list.txt", "r")
digits = [str(i) for i in np.arange(0, 10)]

fonts_list = []
for line in fhandle:
    fonts_list.append(line.rstrip("\n"))

total_fonts = len(fonts_list)
all_fonts = glob.glob("C:\\Windows\\Fonts\\*.ttf")
f_flag = np.zeros(total_fonts)

for sys_font in all_fonts:

    font_file = ntpath.basename(sys_font)
    font_file = font_file.rsplit(".")
    font_file = font_file[0]
    f_idx = 0
    for font in fonts_list:
        f_lower = font.lower()
        s_lower = sys_font.lower()
        # Check desired font
        if f_lower in s_lower:
            path = sys_font
            font = ImageFont.truetype(path, fontSize)
            f_flag[f_idx] = 1
            for ch in digits:
                pos_idx = 0
                for i in [-3, 0, 3]:
                    for j in [-3, 0, 3]:

                        image = Image.new("RGB", imgSize, (0, 0, 0))
                        draw = ImageDraw.Draw(image)

                        style_idx = 0
                        for y in [-1, 0, 1]:
                            for x in [-1, 0, 1]:
                                position = (x + i + init_position[0], y + j + init_position[1])
                                draw.text(position, ch, (255, 255, 255), font=font, align="center")
                                file_name = (
                                    font_file
                                    + "_"
                                    + ch
                                    + "_"
                                    + str(style_idx)
                                    + "_"
                                    + str(pos_idx)
                                    + ".png"
                                )

                                file_name = os.path.join(dataset_path, file_name)
                                image.save(file_name)
                                style_idx += 1
                        pos_idx += 1
        f_idx += 1
