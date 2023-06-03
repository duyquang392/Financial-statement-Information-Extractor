import numpy as np
import time
import json
import os
import cv2
import utility as utility
from utility import get_image_file_list, check_and_read
from src.ocr.data import create_operators, transform
from src.ocr.postprocess import build_post_process


class TextDetector(object):
    def __init__(self, args):
        self.args = args
        self.det_algorithm = args.det_algorithm
        self.predictor, self.input_tensor, self.output_tensors, self.config = utility.create_predictor(
            args, 'det')
        img_h, img_w = self.input_tensor.shape[2:]
        
        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    

if __name__ == "__main__":
    args = utility.parse_args()
    image_file_list = get_image_file_list(args.image_dir)
    text_detector = TextDetector(args)
    total_time = 0
    draw_img_save_dir = args.draw_img_save_dir
    os.makedirs(draw_img_save_dir, exist_ok=True)

    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(2):
            res = text_detector(img)

    save_results = []
    for idx, image_file in enumerate(image_file_list):
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)
        if not flag_pdf:
            if img is None:
                print("error in loading image:{}".format(image_file))
                continue
            imgs = [img]
        else:
            page_num = args.page_num
            if page_num > len(img) or page_num == 0:
                page_num = len(img)
            imgs = img[:page_num]
        for index, img in enumerate(imgs):
            st = time.time()
            dt_boxes, _ = text_detector(img)
            elapse = time.time() - st
            total_time += elapse
            if len(imgs) > 1:
                save_pred = os.path.basename(image_file) + '_' + str(
                    index) + "\t" + str(
                        json.dumps([x.tolist() for x in dt_boxes])) + "\n"
            else:
                save_pred = os.path.basename(image_file) + "\t" + str(
                    json.dumps([x.tolist() for x in dt_boxes])) + "\n"
            save_results.append(save_pred)
            print(save_pred)
            if len(imgs) > 1:
                print("{}_{} The predict time : {}".format(
                    idx, index, elapse))
            else:
                print("{} The predict time : {}".format(
                    idx,  elapse))

            src_im = utility.draw_text_det_res(dt_boxes, img)

            if flag_gif:
                save_file = image_file[:-3] + "png"
            elif flag_pdf:
                save_file = image_file.replace('.pdf',
                                               '_' + str(index) + '.png')
            else:
                save_file = image_file
            img_path = os.path.join(
                draw_img_save_dir,
                "det_res_{}".format(os.path.basename(save_file)))
            cv2.imwrite(img_path, src_im)
            print("The visualized image saved in {}".format(img_path))

    with open(os.path.join(draw_img_save_dir, "det_results.txt"), 'w') as f:
        f.writelines(save_results)
        f.close()
