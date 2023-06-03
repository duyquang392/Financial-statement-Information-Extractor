import os
import cv2
import json
import time
from copy import deepcopy
import utility as utility
from utility import get_image_file_list, check_and_read
from src.ocr.ocr import TextSystem
from src.structure.layout.layout_analysis import LayoutAnalysis
from src.structure.table.predict_table import TableSystem, to_excel
from utility import read_header_tabel_dict


class StructureSystem(object):
    def __init__(self, args):
        # init model
        self.layout_predictor = None
        self.text_system = None
        self.table_system = None
        self.header_table_dict = read_header_tabel_dict(args.header_tabel_dict)
        
        ## yolov7_detector = YOLOv7(model_path, conf_thres=0.35, iou_thres=0.65)
        self.layout_predictor = LayoutAnalysis(args)
        self.text_system = TextSystem(args)
        if self.text_system is not None:
            self.table_system = TableSystem(
                args, self.text_system.text_detector,
                self.text_system.text_recognizer)
        else:
            self.table_system = TableSystem(args)


    def __call__(self, img, pre_page, return_ocr_result_in_table=False, img_idx=0):
        time_dict = {
            'layout': 0,
            'table': 0,
            'table_match': 0,
            'det': 0,
            'rec': 0,
            'all': 0
        }
        start = time.time()
        ori_im = img.copy()
        if self.layout_predictor is not None:
            layout_res, elapse = self.layout_predictor(img, img_idx)
            time_dict['layout'] += elapse
        else:
            h, w = ori_im.shape[:2]
            layout_res = [dict(bbox=None, label='table')]
        res_list = []
        tableextract = False
        table_name = '0'
        have_header = False
       
        if not have_header and img_idx - pre_page['index_pre'] == 1:
            tableextract = True
            table_name = pre_page['table_name']
            


        if tableextract:
            for region in layout_res:
                res = ''
                if region['bbox'] is not None:
                    x1, y1, x2, y2 = region['bbox']
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    roi_img = ori_im[y1:y2, x1:x2, :]
                else:
                    x1, y1, x2, y2 = 0, 0, w, h
                    roi_img = ori_im
                if region['label'] == 'table':
                    pre_page['index_pre'] = img_idx
                    if self.table_system is not None:
                        res, table_time_dict = self.table_system(
                            roi_img, return_ocr_result_in_table)
                        time_dict['table'] += table_time_dict['table']
                        time_dict['table_match'] += table_time_dict['match']
                        time_dict['det'] += table_time_dict['det']
                        time_dict['rec'] += table_time_dict['rec']
                else:
                    if self.text_system is not None:
                        filter_boxes, filter_rec_res, ocr_time_dict = self.text_system(
                            roi_img)
                        time_dict['det'] += ocr_time_dict['det']
                        time_dict['rec'] += ocr_time_dict['rec']

                        # remove style char,
                        # when using the recognition model trained on the PubtabNet dataset,
                        # it will recognize the text format in the table, such as <b>
                        style_token = [
                            '<strike>', '<strike>', '<sup>', '</sub>', '<b>',
                            '</b>', '<sub>', '</sup>', '<overline>',
                            '</overline>', '<underline>', '</underline>', '<i>',
                            '</i>'
                        ]
                        res = []
                        for box, rec_res in zip(filter_boxes, filter_rec_res):
                            rec_str, rec_conf = rec_res
                            for token in style_token:
                                if token in rec_str:
                                    rec_str = rec_str.replace(token, '')
                            
                            box += [x1, y1]
                            res.append({
                                'text': rec_str,
                                'confidence': float(rec_conf),
                                'text_region': box.tolist()
                            })
                
                res_list.append({
                    'type': region['label'].lower(),
                    'bbox': [x1, y1, x2, y2],
                    'img': roi_img,
                    'res': res,
                    'img_idx': img_idx
                })
            tableextract = False
        end = time.time()
        time_dict['all'] = end - start
        # print("table name:", table_name)
        return res_list, time_dict, table_name


def save_structure_res(res, save_folder, img_name, table_name, table_dict, img_idx=0):
    excel_save_folder = os.path.join(save_folder, img_name)
    os.makedirs(excel_save_folder, exist_ok=True)
    res_cp = deepcopy(res)
    # save res
    with open(
            os.path.join(excel_save_folder, 'res_{}.txt'.format(img_idx)),
            'w',
            encoding='utf8') as f:
        for region in res_cp:
            roi_img = region.pop('img')
            f.write('{}\n'.format(json.dumps(region)))

            if region['type'].lower() == 'table' and len(region[
                    'res']) > 0 and 'html' in region['res']:
                excel_path = os.path.join(
                    excel_save_folder,
                    '{}_{}_{}.xlsx'.format(table_name, img_idx, region['bbox']))
                to_excel(region['res']['html'], excel_path)
                if table_name == '1':
                    table_dict["BCDKT"].append(excel_path)
                elif table_name == '2':
                    table_dict["BLCTT"].append(excel_path)
                elif table_name == '3':
                    table_dict["BCKQHDKD"].append(excel_path)
            elif region['type'].lower() == 'figure':
                img_path = os.path.join(
                    excel_save_folder,
                    '{}_{}_{}.jpg'.format(table_name, img_idx, region['bbox']))
                cv2.imwrite(img_path, roi_img)


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list
    image_file_list = image_file_list[args.process_id::args.total_process_num]

    if not args.use_pdf2docx_api:
        structure_sys = StructureSystem(args)
        save_folder = os.path.join(args.output, structure_sys.mode)
        os.makedirs(save_folder, exist_ok=True)
    img_num = len(image_file_list)

    for i, image_file in enumerate(image_file_list):
        print("[{}/{}] {}".format(i, img_num, image_file))
        img, flag_gif, flag_pdf = check_and_read(image_file)
        img_name = os.path.basename(image_file).split('.')[0]
        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)

        if not flag_pdf:
            if img is None:
                print("error in loading image:{}".format(image_file))
                continue
            imgs = [img]
        else:
            imgs = img

        all_res = []
        for index, img in enumerate(imgs):
            res, time_dict = structure_sys(img, img_idx=index)
            img_save_path = os.path.join(save_folder, img_name,
                                         'show_{}.jpg'.format(index))
            os.makedirs(os.path.join(save_folder, img_name), exist_ok=True)
            draw_img = utility.draw_structure_result(img, res, args.vis_font_path)
            save_structure_res(res, save_folder, img_name, index)
            if res != []:
                cv2.imwrite(img_save_path, draw_img)
                print('result save to {}'.format(img_save_path))

        print("Predict time : {:.3f}s".format(time_dict['all']))


if __name__ == "__main__":
    args = utility.parse_args()
    main(args)
