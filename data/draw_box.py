import os
import random
import sys
import getopt
import logging
import cv2


logger = logging.getLogger('TinyML')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)



def draw_box(input_file, result_file):
    font_scale = 0.4
    text_thickness = 1
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    box_value = []

    with open(result_file, "r") as f:
        for line in f.readlines():
            line = line.strip('/n')
            temp = line.split()
            box_value.append(temp)

    num = int(box_value[0][2])
    color = []
    while len(color) != num:
        r = random.randint(0, 256)
        g = random.randint(0, 256)
        b = random.randint(0, 256)
        if (r, g, b) not in color:
            color.append((r, g, b))

    result_img_w = int(box_value[0][0])
    result_img_h = int(box_value[0][1])
    img = cv2.imread(input_file)
    img_ori_h, img_ori_w = img.shape[:2]
    if result_img_w != img_ori_w or result_img_h != img_ori_h:
        img = cv2.resize(img, (result_img_w, result_img_h))
    
    for i in range(1, len(box_value)):
        cls_id = int(box_value[i][1])
        txt = "{}: {}".format(box_value[i][0], box_value[i][2])
        txt_loc = (int(float(box_value[i][3])) + 2, int(float(box_value[i][4])) + 1)
        cv2.putText(img, txt, txt_loc, font_face, font_scale, (0, 0, 255), text_thickness, 8)
        
        left_up = (int(float(box_value[i][3])), int(float(box_value[i][4])))
        right_down = (int(float(box_value[i][5])), int(float(box_value[i][6])))
        cv2.rectangle(img, left_up, right_down, color[cls_id], text_thickness)   
  
    out_image_path = "output_" + os.path.basename(input_file)
    cv2.imwrite(out_image_path, img)
    logger.info("image {} write success.".format(input_file))
    

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:r:", ["ifile=", "rfile="])
    except getopt.GetoptError:
        logger.error("draw_box.py -i <image_path> -r <result_path>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-i":
            image_path = arg
        elif opt == "-r":
            result_path = arg
        elif opt == "-n":
            classes_num = arg
        else:
            logger.error("draw_box.py -i <image_path> -r <result_path>")

    draw_box(image_path, result_path)