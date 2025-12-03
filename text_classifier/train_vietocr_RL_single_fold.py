import os
import yaml
import time
from vietocr.vietocr.tool.config import Cfg
from vietocr.vietocr.model.trainer_RL import TrainerRL
import shutil
from vietocr.vietocr.tool.create_dataset import createDataset

# === CONFIGURATION ===
base_config_path = "mc_ocr/text_classifier/vietocr/config/base.yml"        
seq2seq_config_path = "mc_ocr/text_classifier/vietocr/config/vgg-seq2seq.yml"  
folds_root = "mc_ocr/data/folds"              
log_file_path = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\apply_RL\eval_summary_log.txt"

def update_fold_config(base_yaml_path, output_yaml_path, fold_path):
    with open(base_yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config['dataset']['data_root'] = os.path.join(fold_path, 'training_ocr_data/train/images')
    config['dataset']['train_annotation'] = os.path.join(fold_path, 'training_ocr_data/train/train_ocr_fixed.txt')
    config['dataset']['valid_data_root'] = os.path.join(fold_path, 'training_ocr_data/val/images')
    config['dataset']['valid_annotation'] = os.path.join(fold_path, 'training_ocr_data/val/val_ocr_fixed.txt')

    config['trainer']['export'] = os.path.join(fold_path, 'model.pth')
    config['trainer']['checkpoint'] = os.path.join(fold_path, 'checkpoint.pth')
    config['trainer']['log'] = os.path.join(fold_path, 'train.log')

    with open(output_yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f)

# === TRAINING FOR FOLD TESTING ===
start_all = time.time()
fold = "test"

fold_path = os.path.join(folds_root, f"fold_{fold}")
updated_yaml_path = f"mc_ocr/text_classifier/vietocr/config/base_fold{fold}.yml"

update_fold_config(base_config_path, updated_yaml_path, fold_path)

config = Cfg.load_config(updated_yaml_path, seq2seq_config_path)
trainer = TrainerRL(config)

num_epochs = 3
for epoch in range(num_epochs):
    for batch in trainer.train_gen:
        loss = trainer.step(batch)
        print(f"[Fold {fold}] Epoch {epoch+1} - Train Loss: {loss:.4f}")

end_all = time.time()
total_duration = end_all - start_all
print(f"\n===== Total Training Time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes) =====")
