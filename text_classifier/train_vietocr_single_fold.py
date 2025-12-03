import os
import yaml
import time
from vietocr.vietocr.tool.config import Cfg
from vietocr.vietocr.model.trainer import Trainer
import shutil
from vietocr.vietocr.tool.create_dataset import createDataset

# === CONFIGURATION ===
base_config_path = "mc_ocr/text_classifier/vietocr/config/base.yml"        
seq2seq_config_path = "mc_ocr/text_classifier/vietocr/config/vgg-seq2seq.yml"  
transformer_config_path = "mc_ocr/text_classifier/vietocr/config/vgg-transformer.yml"  
folds_root = "mc_ocr/data/folds"              
log_file_path = r"C:\Users\hongt\OneDrive\Desktop\Thesis\MC_OCR\mc_ocr\apply_RL\fold_2_log.txt"

def update_fold_config(base_yaml_path, output_yaml_path, fold_path):
    with open(base_yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config['dataset']['data_root'] = fold_path
    config['dataset']['train_data_root'] = os.path.join(fold_path, 'training_ocr_data/train/images')
    config['dataset']['valid_data_root'] = os.path.join(fold_path, 'training_ocr_data/val/images')

    config['dataset']['train_annotation'] = os.path.join(fold_path, 'training_ocr_data/train/train_ocr_fixed.txt')
    config['dataset']['valid_annotation'] = os.path.join(fold_path, 'training_ocr_data/val/val_ocr_fixed.txt')

    # config['trainer']['export'] = os.path.join(fold_path, 'model.pth')
    config['trainer']['checkpoint'] = os.path.join(fold_path, 'checkpoint.pth')
    config['trainer']['log'] = os.path.join(fold_path, 'train.log')
    
    with open(output_yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f)

    # print("\n=== Configuration Loaded ===")
    # print(f"Train Data Root: {config['dataset']['data_root']}")
    # print(f"Train Annotation: {config['dataset']['train_annotation']}")
    # print(f"Valid Data Root: {config['dataset']['valid_data_root']}")
    # print(f"Valid Annotation: {config['dataset']['valid_annotation']}")

# === TẠO LMDB CHO TRAIN VÀ VAL ===
def create_lmdb_for_train_and_val(fold_path):
    train_lmdb_path = os.path.join(fold_path, 'training_ocr_data/train/train_data')
    val_lmdb_path = os.path.join(fold_path, 'training_ocr_data/val/val_data')

    # Chỉ tạo lại LMDB nếu chưa tồn tại
    if not os.path.exists(train_lmdb_path):
        print(f"\nCreating LMDB for TRAIN: {train_lmdb_path}...")
        createDataset(
            outputPath=train_lmdb_path,
            root_dir=os.path.join(fold_path, 'training_ocr_data/train/images'),
            annotation_path=os.path.join(fold_path, 'training_ocr_data/train/train_ocr_fixed.txt')
        )
    else:
        print(f"LMDB for TRAIN already exists: {train_lmdb_path}")

    if not os.path.exists(val_lmdb_path):
        print(f"\nCreating LMDB for VALIDATION: {val_lmdb_path}...")
        createDataset(
            outputPath=val_lmdb_path,
            root_dir=os.path.join(fold_path, 'training_ocr_data/val/images'),
            annotation_path=os.path.join(fold_path, 'training_ocr_data/val/val_ocr_fixed.txt')
        )
    else:
        print(f"LMDB for VALIDATION already exists: {val_lmdb_path}")

# === TRAINING FOR FOLD TESTING ===
# start_all = time.time()
fold = "2"

fold_path = os.path.join(folds_root, f"fold_{fold}")
updated_yaml_path = f"mc_ocr/text_classifier/vietocr/config/base_fold_{fold}.yml"

update_fold_config(base_config_path, updated_yaml_path, fold_path)
create_lmdb_for_train_and_val(fold_path)

print(f"\nTraining Fold {fold}...")

# Load config and train using Trainer API directly
config = Cfg.load_config(updated_yaml_path, seq2seq_config_path)
trainer = Trainer(config)

def custom_train(trainer):
    # with open(log_file_path, 'a', encoding='utf-8') as log_file:
    #     best_acc = 0
    #     num_epochs = 3 

    #     for epoch in range(1, num_epochs + 1):
    #         trainer.model.train()
    #         epoch_loss = 0.0

    #         print(f"Trainer Train Gen: {trainer.train_gen}")
    #         print(f"Type of Trainer Train Gen: {type(trainer.train_gen)}")

    #         if trainer.train_gen is None:
    #             print("Trainer train_gen is None. Please check your data loading configuration.")
    #             return

    #         for i, batch in enumerate(trainer.train_gen):
    #             print(f"Batch {i+1}:", batch)
    #             loss = trainer.step(batch)
    #             epoch_loss += loss
            
    #         avg_loss = epoch_loss / len(trainer.train_gen)
    #         print(f"[Fold {fold}] Epoch {epoch} - Train Loss: {avg_loss:.4f}")
    #         log_file.write(f"[Fold {fold}] Epoch {epoch} - Train Loss: {avg_loss:.4f}\n")

    #         val_loss = trainer.validate()
    #         acc_full_seq, acc_per_char = trainer.precision()
    #         print(f"[Fold {fold}] Epoch {epoch} - Val Loss: {val_loss:.4f} | Acc Full Seq: {acc_full_seq:.4f} | Acc Per Char: {acc_per_char:.4f}")
    #         log_file.write(f"[Fold {fold}] Epoch {epoch} - Val Loss: {val_loss:.4f} | Acc Full Seq: {acc_full_seq:.4f} | Acc Per Char: {acc_per_char:.4f}\n")

    #         if acc_full_seq > best_acc:
    #             best_acc = acc_full_seq
    #             trainer.save_weights(config['trainer']['export'])
    #             print(f"Model improved, saved to {config['trainer']['export']}")
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        best_cer = float('inf') 
        num_epochs = 1

        for epoch in range(1, num_epochs + 1):
            start_train = time.time()
            trainer.model.train()
            epoch_loss = 0.0
            total_cer = 0.0
            total_samples = 0

            # print(f"Trainer Train Gen: {trainer.train_gen}")
            # print(f"Type of Trainer Train Gen: {type(trainer.train_gen)}")

            # if trainer.train_gen is None:
            #     print("Trainer train_gen is None. Please check your data loading configuration.")
            #     return

            for i, batch in enumerate(trainer.train_gen):
                # print(f"Batch {i+1}:", batch)
                loss = trainer.step(batch)
                epoch_loss += loss

                # Tính CER cho từng batch
                batch_cer = trainer.calculate_batch_cer(batch)
                total_cer += batch_cer
                total_samples += 1
            
            avg_loss = epoch_loss / len(trainer.train_gen)
            avg_cer = total_cer / total_samples

            end_train = time.time()
            train_duration = end_train - start_train

            print(f"[Fold {fold}] Epoch {epoch} - Train Loss: {avg_loss:.4f} | Train CER: {avg_cer:.4f} | Duration: {train_duration:.2f} seconds", flush = True)
            log_file.write(f"[Fold {fold}] Epoch {epoch} - Train Loss: {avg_loss:.4f} | Train CER: {avg_cer:.4f} | Duration: {train_duration:.2f} seconds\n")
            log_file.flush()

            start_val = time.time()
            val_loss, val_cer = trainer.validate()
            acc_full_seq, acc_per_char = trainer.precision()
            end_val = time.time()
            val_duration = end_val - start_val

            print(f"[Fold {fold}] Epoch {epoch} - Val Loss: {val_loss:.4f} | Val CER: {val_cer:.4f} | Acc Full Seq: {acc_full_seq:.4f} | Acc Per Char: {acc_per_char:.4f} | Duration: {val_duration:.2f} seconds", flush = True)
            log_file.write(f"[Fold {fold}] Epoch {epoch} - Val Loss: {val_loss:.4f} | Val CER: {val_cer:.4f} | Acc Full Seq: {acc_full_seq:.4f} | Acc Per Char: {acc_per_char:.4f} | Duration: {val_duration:.2f} seconds\n")
            log_file.flush()

            if val_cer < best_cer:
                best_cer = val_cer
                # trainer.save_weights(config['trainer']['export'])
                best_model_path = os.path.join(fold_path, f"best_model_epoch_{epoch}.pth")
                trainer.save_weights(best_model_path)
                print(f"Model improved at epoch {epoch}, saved to {config['trainer']['export']}")
                log_file.write(f"Model improved at epoch {epoch}, saved to {config['trainer']['export']}\n")
                log_file.flush()
            
custom_train(trainer)
