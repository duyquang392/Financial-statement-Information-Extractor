import cv2
import os
from src.structure.predict_system import StructureSystem, save_structure_res
import src.ocr.predict_cls as classifier
import src.ocr.postprocess as config
from utility import get_image_file_list, check_and_read, concat_excel2html, empty_output, empty_output_folder
from utility import parse_args, draw_structure_result
from argparse import ArgumentParser
import yaml
from yaml.loader import SafeLoader
from distutils.log import debug
import time
from flask import *  

app = Flask(__name__)


def run(image_dir):
    start_time = time.time()
    empty_output_folder("output/layout")
    image_file_list = get_image_file_list(image_dir)
    image_file_list = image_file_list
    image_file_list = image_file_list[process_id::total_process_num]
    
    # text_classifier = classifier.TextClassifier(args)

    # structure_sys = StructureSystem(args)
    # save_folder = os.path.join(args.output, 'structure')
    # os.makedirs(save_folder, exist_ok=True)
    # img_num = len(image_file_list)

    for i, image_file in enumerate(image_file_list):
        table_dict = {
        "BCDKT" : [],
        "BLCTT" : [],
        "BCKQHDKD" : []
        }
        print("[{}/{}] {}".format(i+1, img_num, image_file))
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
        pre_page ={
            'index_pre' : 10000,
            'table_name' : ''
        }
        for index, img in enumerate(imgs):
            if index < 12:
                print('index: ', index)
                img, angle_list, elapse = text_classifier([img])     
                res, time_dict, table_name = structure_sys(img, pre_page, img_idx=index)
                img_save_path = os.path.join(save_folder, img_name,
                                            'show_{}.jpg'.format(index))
                os.makedirs(os.path.join(save_folder, img_name), exist_ok=True)
                if res != []:
                    draw_img = draw_structure_result(img, res, './src/utils/simfang.ttf')
                    save_structure_res(res, save_folder, img_name, table_name, table_dict, index)
                
                if res != []:
                    cv2.imwrite(img_save_path, draw_img)
            else:
                print('index: ', index)
        concat_excel2html( table_dict)
        print('time pipeline: ', time.time() - start_time)

@app.route('/')  
def main():  
    return render_template("index.html")

@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST': 
        empty_output()
        f = request.files['filequery']
        f.save('input/' + f.filename)
        run('input/' + f.filename)
        os.remove('input/' + f.filename)
    return render_template("output.html") 

@app.route('/download/<name>', methods = ['GET','POST']) 
def download_file(name):
    return send_from_directory(app.config["OUTPUT"], name)

app.add_url_rule(
    "/download/<name>", endpoint="download_file", build_only=True
)

if __name__ == "__main__":
    args = parse_args()
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list
    process_id = args.process_id
    total_process_num = args.total_process_num
    empty_output()
    # args_cls = config.parse_args()
    # config = config.get_config(args_cls.config, overrides=args_cls.override, show=True)
    text_classifier = classifier.TextClassifier(args)

    structure_sys = StructureSystem(args)
    
    save_folder = os.path.join(args.output, 'structure')
    os.makedirs(save_folder, exist_ok=True)
    img_num = len(image_file_list)
    app.run(host = '127.0.0.6', port = '8686', debug = True)


    run(args)

