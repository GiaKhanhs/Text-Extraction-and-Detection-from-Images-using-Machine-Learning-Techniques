import numpy as np
import cv2, os, time
from datetime import datetime
from mc_ocr.utils.common import get_list_file_in_folder
from mc_ocr.utils.visualize import viz_icdar
from mc_ocr.rotation_corrector.predict import init_box_rectify_model
from mc_ocr.rotation_corrector.utils.utils import rotate_image_bbox_angle
from mc_ocr.rotation_corrector.filter import drop_box, get_mean_horizontal_angle, filter_90_box
from mc_ocr.rotation_corrector.utils.line_angle_correction import rotate_and_crop
from mc_ocr.config import rot_drop_thresh, rot_visualize, rot_model_path

os.environ["PYTHONIOENCODING"] = "utf-8"
pred_time = datetime.today().strftime('%Y-%m-%d_%H-%M')

gpu = '0'

crop_method = 2  # overall method 2 is better than method 1 FOR CLASSIFY
classifier_batch_sz = 4
worker = 1
write_rotated_img = True
write_file = True
visualize = rot_visualize
# extend_bbox = True  # extend bbox when crop or not
debug = False

# box rotation classifier
weight_path = rot_model_path
classList = ['0', '180']

if gpu is None or debug:
    classifier_batch_sz = 1
    worker = 0


def write_output(list_boxes, result_file_path):
    result = ''
    for idx, box_data in enumerate(list_boxes):
        if isinstance(box_data, dict):
            box = box_data['coors']
            s = [str(i) for i in box]
            line = ','.join(s) + box_data['data']
        else:
            box = box_data
            s = [str(i) for i in box]
            line = ','.join(s) + ','
        result += line + '\n'
    result = result.rstrip('\n')
    with open(result_file_path, 'w', encoding='utf8') as res:
        res.write(result)


def get_boxes_data(img_data, boxes):
    boxes_data = []
    for box_data in boxes:
        if isinstance(box_data, dict):
            box_loc = box_data['coors']
        else:
            box_loc = box_data
        box_loc = np.array(box_loc).astype(np.int32).reshape(-1, 1, 2)
        box_data = rotate_and_crop(img_data, box_loc, debug=False, extend=True,
                                   extend_x_ratio=0.0001,
                                   extend_y_ratio=0.0001,
                                   min_extend_y=2, min_extend_x=1)

        boxes_data.append(box_data)
    return boxes_data


def calculate_page_orient(box_rectify, img_rotated, boxes_list):
    boxes_data = get_boxes_data(img_rotated, boxes_list)
    rotation_state = {'0': 0, '180': 0}
    for it, img in enumerate(boxes_data):
        _, degr = box_rectify.inference(img, debug=False)
        rotation_state[degr[0]] += 1
    print(rotation_state)
    if rotation_state['0'] >= rotation_state['180']:
        ret = 0
    else:
        ret = 180
    return ret


def main_rotation_corrector(image_names, img_dir, anno_dir,
                            output_rotated_img_dir, output_txt_dir, output_viz_dir):
    global anno_path

    box_rectify = init_box_rectify_model(rot_model_path)

    list_img_path = [os.path.join(img_dir, name) for name in image_names]

    for idx, img_path in enumerate(list_img_path):
        print('\n', idx, 'Inference', img_path)
        test_img = cv2.imread(img_path)
        if test_img is None:
            print(f"[Warning] Cannot read image: {img_path}")
            continue

        img_name = os.path.basename(img_path)
        anno_path = os.path.join(anno_dir, img_name.replace('.jpg', '.txt'))
        if not os.path.exists(anno_path):
            print(f"[Warning] Missing annotation: {anno_path}")
            continue

        boxes_list = get_list_boxes_from_icdar(anno_path)
        boxes_list = drop_box(boxes_list, drop_gap=rot_drop_thresh)
        if not boxes_list or len(boxes_list) == 0:
            print(f"[Info] No valid boxes in {img_path}, saving original image...")
            # Lưu ảnh gốc (không xoay)
            output_rotated_img_path = os.path.join(output_rotated_img_dir, img_name)
            cv2.imwrite(output_rotated_img_path, test_img)

            # Ghi file txt trống
            output_txt_path = os.path.join(output_txt_dir, img_name.replace('.jpg', '.txt'))
            with open(output_txt_path, 'w', encoding='utf8') as f:
                pass  # ghi file rỗng

            # Visualize (có hoặc không)
            if visualize:
                output_viz_path = os.path.join(output_viz_dir, img_name)
                cv2.imwrite(output_viz_path, test_img)

            continue  # skip xoay
        rotation = get_mean_horizontal_angle(boxes_list, False)
        img_rotated, boxes_list = rotate_image_bbox_angle(test_img, boxes_list, rotation)

        degre = calculate_page_orient(box_rectify, img_rotated, boxes_list)
        img_rotated, boxes_list = rotate_image_bbox_angle(img_rotated, boxes_list, degre)
        boxes_list = filter_90_box(boxes_list)

        output_txt_path = os.path.join(output_txt_dir, img_name.replace('.jpg', '.txt'))
        if os.path.exists(output_txt_path):
            print(f"[Skip] Already processed: {img_path}")
            continue
        output_viz_path = os.path.join(output_viz_dir, img_name)
        if os.path.exists(output_viz_path):
            print(f"[Skip] Already processed: {img_path}")
            continue
        output_rotated_img_path = os.path.join(output_rotated_img_dir, img_name)
        if os.path.exists( output_rotated_img_path):
            print(f"[Skip] Already processed: {img_path}")
            continue

        if write_rotated_img:
            cv2.imwrite(output_rotated_img_path, img_rotated)
        if write_file:
            write_output(boxes_list, output_txt_path)
        if visualize:
            viz_icdar(img_rotated, output_txt_path, output_viz_path)


def get_list_boxes_from_icdar(anno_path):
    with open(anno_path, 'r', encoding='utf-8') as f:
        anno_txt = f.readlines()
    list_boxes = []
    for anno in anno_txt:
        anno = anno.rstrip('\n')

        idx = -1
        for i in range(0, 8):
            idx = anno.find(',', idx + 1)

        coordinates = anno[:idx]
        coors = [int(f) for f in coordinates.split(',')]
        list_boxes.append({'coors': coors, 'data': anno[idx:]})
    return list_boxes


if __name__ == '__main__':
    base_fold_dir = "mc_ocr/data/folds"
    raw_img_root = "mc_ocr/data/train_images"

    for fold in range(2, 3):
        # for split in ["val"]:
        for split in ["train", "val"]:
            print(f"\n=== Fold {fold} | Split: {split} ===")

            fold_dir = os.path.join(base_fold_dir, f"fold_{fold}")
            split_txt_file = os.path.join(fold_dir, f"{split}.txt")
            with open(split_txt_file, "r", encoding="utf-8") as f:
                image_list = [line.strip() for line in f.readlines()]
            
            img_dir = raw_img_root
            anno_dir = os.path.join(fold_dir, "outputs", f"det_txt_{split}")

            output_base = os.path.join(fold_dir, "outputs")
            rot_out_img_dir = os.path.join(output_base, f"rot_out_{split}_img")
            rot_out_txt_dir = os.path.join(output_base, f"rot_out_{split}_txt")
            rot_out_viz_dir = os.path.join(output_base, f"rot_out_{split}_viz")

            os.makedirs(rot_out_img_dir, exist_ok=True)
            os.makedirs(rot_out_txt_dir, exist_ok=True)
            os.makedirs(rot_out_viz_dir, exist_ok=True)

            main_rotation_corrector(image_list, img_dir, anno_dir,
                                    rot_out_img_dir, rot_out_txt_dir, rot_out_viz_dir)
