import os

CONFIG_ROOT = os.path.dirname(__file__)
OUTPUT_ROOT = 'C:/Users/hongt/OneDrive/Desktop/Thesis/MC_OCR/mc_ocr/data'


def full_path(sub_path, file=False):
    path = os.path.join(CONFIG_ROOT, sub_path)
    if not file and not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            print('full_path. Error makedirs',path)
    return path


def output_path(sub_path):
    path = os.path.join(OUTPUT_ROOT, sub_path)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            print('output_path. Error makedirs',path)
    return path

# gpu = '0'  # None or 0,1,2...
gpu = None
dataset = 'mcocr_private_test_data'
# dataset = "inference"
# mcocr_val_data
# mc_ocr_train_filtered
# 'mc_ocr_train'

# input data from organizer
raw_train_img_dir = full_path(r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\inference\image")
raw_img_dir = full_path(r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\inference\image")

# raw_train_img_dir = full_path('data/mcocr_private_test_data/test_images')
# raw_img_dir = full_path('data/mcocr_private_test_data/test_images')
# raw_train_img_dir = full_path('data/mcocr_val_data/val_images')
# raw_img_dir = full_path('data/mcocr_val_data/val_images')
# raw_img_dir=full_path('data/{}'.format(dataset))

raw_csv = full_path('data/mcocr_train_df.csv', file=True)

# EDA
json_data_path = full_path('EDA/final_data.json', file=True)
filtered_train_img_dir=full_path('data/mc_ocr_train_filtered')
filtered_csv = full_path('data/mcocr_train_df_filtered.csv', file=True)

# text detector
det_model_dir = full_path('text_detector/PaddleOCR/inference/ch_ppocr_server_v2.0_det_infer')
det_visualize = True
det_db_thresh = 0.3
det_db_box_thresh = 0.3

det_out_viz_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\inference\text_detector/viz_imgs"
det_out_txt_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\inference\text_detector/txt"

# det_out_viz_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_private_test_data\text_detector/viz_imgs"
# det_out_txt_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_private_test_data\text_detector/txt"
# det_out_viz_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_val_data/text_detector/viz_imgs"
# det_out_txt_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_val_data/text_detector/txt"
# det_out_viz_dir = output_path('text_detector/{}/viz_imgs'.format(dataset))
# det_out_txt_dir = output_path('text_detector/{}/txt'.format(dataset))

# C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_val_data

# rotation corrector
rot_drop_thresh = [.5, 2]
rot_visualize = True
rot_model_path = full_path('rotation_corrector/weights/mobilenetv3-Epoch-487-Loss-0.03-Acc-0.99.pth', file=True)
rot_out_img_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\inference/rotation_corrector/imgs"
rot_out_txt_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\inference/rotation_corrector/txt"
rot_out_viz_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\inference/rotation_corrector/viz_imgs"
# rot_out_img_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_val_data/rotation_corrector/imgs"
# rot_out_txt_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_val_data/rotation_corrector/txt"
# rot_out_viz_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_val_data/rotation_corrector/viz_imgs"
# rot_out_img_dir = output_path('rotation_corrector/{}/imgs'.format(dataset))
# rot_out_txt_dir = output_path('rotation_corrector/{}/txt'.format(dataset))
# rot_out_viz_dir = output_path('rotation_corrector/{}/viz_imgs'.format(dataset))
rotate_filtered_csv = full_path('data/mcocr_train_df_rotate_filtered.csv', file=True)

# text classifier (OCR)
cls_visualize = True
cls_ocr_thres = 0.65
# cls_model_path = full_path('text_classifier/vietocr/vietocr/weights/best_model_epoch_11.pth', file=True)
cls_model_path = full_path('text_classifier/vietocr/vietocr/weights/best_model_epoch_5.pth', file=True)
cls_base_config_path = full_path('text_classifier/vietocr/config/base.yml', file=True)
cls_config_path = full_path('text_classifier/vietocr/config/vgg-seq2seq.yml', file=True)
cls_out_viz_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\inference/text_classifier/viz_imgs"
cls_out_txt_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\inference/text_classifier/txt"
# cls_out_viz_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_private_test_data/text_classifier/viz_imgs"
# cls_out_txt_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_private_test_data/text_classifier/txt"
# cls_out_viz_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_val_data/text_classifier/viz_imgs"
# cls_out_txt_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_val_data/text_classifier/txt"
# cls_out_viz_dir = output_path('text_classifier/{}/viz_imgs'.format(dataset))
# cls_out_txt_dir = output_path('text_classifier/{}/txt'.format(dataset))

# key information
kie_visualize = True
kie_model = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\key_info_extraction\model_best.pth"
kie_boxes_transcripts = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_private_test_data\key_info_extraction/boxes_and_transcripts"
kie_out_txt_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_private_test_data\key_info_extraction/txt"
kie_out_viz_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_private_test_data\key_info_extraction/viz_imgs"
# kie_boxes_transcripts = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_private_test_data\key_info_extraction/boxes_and_transcripts"
# kie_out_txt_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_private_test_data\key_info_extraction/txt"
# kie_out_viz_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_private_test_data\key_info_extraction/viz_imgs"
# kie_boxes_transcripts = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_val_data\key_info_extraction/boxes_and_transcripts"
# kie_out_txt_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_val_data\key_info_extraction/txt"
# kie_out_viz_dir = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_val_data\key_info_extraction/viz_imgs"
# kie_boxes_transcripts = output_path('key_info_extraction/{}/boxes_and_transcripts'.format(dataset))
# kie_out_txt_dir = output_path('key_info_extraction/{}/txt'.format(dataset))
# kie_out_viz_dir = output_path('key_info_extraction/{}/viz_imgs'.format(dataset))

# submision
# best_task1_csv = full_path('submit/{}/best_task1_0.11609/results.csv'.format(dataset), file=True)
best_task1_csv = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\submit\mc_ocr_private_test\best_task1_0.11609/results.csv"
submit_sample_file = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_private_test_data\mcocr_test_samples_df.csv"
# submit_sample_file = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\data\mcocr_val_data\mcocr_val_sample_df.csv"
output_submission_file = r"C:\Users\hongt\OneDrive\Desktop\Thesis\result_private_test_data\results.csv"
