import os
import yaml
import time
from vietocr.vietocr.tool.config import Cfg
from vietocr.vietocr.model.trainer import Trainer
import shutil

# === CONFIGURATION ===
base_config_path = "mc_ocr/text_classifier/vietocr/config/base.yml"        
seq2seq_config_path = "mc_ocr/text_classifier/vietocr/config/vgg-seq2seq.yml"  
folds_root = "mc_ocr/data/folds"              
log_file_path = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\apply_RL\eval_summary_log.txt"

# === FUNCTION TO UPDATE CONFIG ===
def update_fold_config(base_yaml_path, output_yaml_path, fold_path):
    with open(base_yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Dữ liệu huấn luyện trong: fold_X/training_ocr_data/train/
    config['dataset']['data_root'] = os.path.join(fold_path, 'training_ocr_data/train')
    config['dataset']['train_annotation'] = 'train_ocr.txt'

    # Dùng đường dẫn tuyệt đối để tránh lỗi khi join từ data_root
    val_annotation_abs_path = os.path.abspath(os.path.join(fold_path, 'training_ocr_data/val/val_ocr.txt'))
    config['dataset']['valid_annotation'] = val_annotation_abs_path

    config['trainer']['export'] = os.path.join(fold_path, 'model.pth')
    config['trainer']['checkpoint'] = os.path.join(fold_path, 'checkpoint.pth')
    config['trainer']['log'] = os.path.join(fold_path, 'train.log')
    
    with open(output_yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f)

# === TRAINING LOOP FOR EACH FOLD ===
start_all = time.time()

for fold in range(3):
    fold_path = os.path.join(folds_root, f"fold_{fold}")
    updated_yaml_path = f"mc_ocr/text_classifier/vietocr/config/base_fold{fold}.yml"

    lmdb_path = os.path.join(fold_path, 'training_ocr_data/train/train_data')
    if os.path.exists(lmdb_path):
        print(f"Removing old LMDB at {lmdb_path}")
        shutil.rmtree(lmdb_path)
    update_fold_config(base_config_path, updated_yaml_path, fold_path)

    print(f"\nTraining Fold {fold}...")

    # Load config and train using Trainer API directly
    config = Cfg.load_config(updated_yaml_path, seq2seq_config_path)
    trainer = Trainer(config)

    def custom_train(trainer):
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            train_dataset_size = len(trainer.train_loader.dataset)
            batch_size = trainer.config['trainer']['batch_size']
            steps_per_epoch = train_dataset_size // batch_size + int(train_dataset_size % batch_size != 0)
            num_epochs = 10

            start_time = time.time()
            log_file.write(f"===== Fold {fold} Training Start: {time.ctime(start_time)} =====\n")

            for epoch in range(1, num_epochs + 1):
                trainer.model.train()
                epoch_loss = 0.0
                for step in range(steps_per_epoch):
                    loss = trainer.train_batch()
                    epoch_loss += loss
                avg_loss = epoch_loss / steps_per_epoch
                msg = f"[Fold {fold}] Epoch {epoch} - Train Loss: {avg_loss:.4f}"
                print(msg)
                log_file.write(msg + "\n")

                val_loss, acc = trainer.validate_batch()
                msg = f"[Fold {fold}] Epoch {epoch} - Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f}"
                print(msg)
                log_file.write(msg + "\n")

            end_time = time.time()
            duration = end_time - start_time
            log_file.write(f"===== Fold {fold} Training End: {time.ctime(end_time)} | Duration: {duration:.2f} seconds =====\n\n")

    custom_train(trainer)

    # SAVE MODEL
    trainer.save_checkpoint(config['trainer']['export'])
    print(f"Saved model for Fold {fold} to {config['trainer']['export']}")

end_all = time.time()
total_duration = end_all - start_all

with open(log_file_path, 'a', encoding='utf-8') as log_file:
    log_file.write(f"\n===== Total Training Time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes) =====\n")
