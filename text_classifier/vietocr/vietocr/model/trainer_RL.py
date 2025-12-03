import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from vietocr.vietocr.tool.translate import build_model
from vietocr.vietocr.tool.utils import download_weights, compute_accuracy
from vietocr.vietocr.loader.dataloader import OCRDataset, ClusterRandomSampler, Collator
from torch.utils.data import DataLoader

class TrainerRL():
    def __init__(self, config, pretrained=True, augmentor=None):
        self.config = config
        self.model, self.vocab = build_model(config)
        self.device = config['device']
        self.num_iters = config['trainer']['iters']
        self.data_root = config['dataset']['data_root']
        self.train_annotation = config['dataset']['train_annotation']
        self.valid_annotation = config['dataset']['valid_annotation']
        self.batch_size = config['trainer']['batch_size']

        self.optimizer = AdamW(self.model.parameters(), lr=0.0003)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=0.0003, total_steps=self.num_iters)

        if pretrained:
            weight_file = download_weights(**config['pretrain'], quiet=config['quiet'])
            self.load_weights(weight_file)

        self.criterion = nn.CrossEntropyLoss()
        self.baseline = 0
        self.reward_history = []
        self.baseline_history = []
        self.loss_history = []

    def step(self, batch):
        self.model.train()

        batch = self.batch_to_device(batch)
        img, tgt_input, tgt_output = batch['img'], batch['tgt_input'], batch['tgt_output']

        outputs = self.model(img, tgt_input)
        log_probs = torch.log_softmax(outputs, dim=-1)

        rewards = self.compute_rewards(log_probs, tgt_output)
        baseline_adjusted_rewards = rewards - self.baseline

        loss = -(log_probs * baseline_adjusted_rewards.unsqueeze(-1)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.baseline = 0.9 * self.baseline + 0.1 * rewards.mean().item()

        # Tracking metrics
        self.reward_history.append(rewards.mean().item())
        self.baseline_history.append(self.baseline)
        self.loss_history.append(loss.item())

        return loss.item()

    def compute_rewards(self, log_probs, tgt_output):
        rewards = (log_probs.argmax(dim=-1) == tgt_output).float()
        return rewards

    def data_gen(self, lmdb_path, data_root, annotation, masked_language_model=True, transform=None):
        dataset = OCRDataset(lmdb_path=lmdb_path, 
                root_dir=data_root, annotation_path=annotation, 
                vocab=self.vocab, transform=transform, 
                image_height=self.config['dataset']['image_height'], 
                image_min_width=self.config['dataset']['image_min_width'], 
                image_max_width=self.config['dataset']['image_max_width'])

        sampler = ClusterRandomSampler(dataset, self.batch_size, True)
        collate_fn = Collator(masked_language_model)

        gen = DataLoader(
                dataset,
                batch_size=self.batch_size, 
                sampler=sampler,
                collate_fn = collate_fn,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                )
       
        return gen

    def data_gen_v1(self, lmdb_path, data_root, annotation):
        data_gen = DataGen(data_root, annotation, self.vocab, 'cpu', 
                image_height = self.config['dataset']['image_height'],        
                image_min_width = self.config['dataset']['image_min_width'],
                image_max_width = self.config['dataset']['image_max_width'])

        return data_gen

    def batch_to_device(self, batch):
        img = batch['img'].to(self.device)
        tgt_input = batch['tgt_input'].to(self.device)
        tgt_output = batch['tgt_output'].to(self.device)

        return {'img': img, 'tgt_input': tgt_input, 'tgt_output': tgt_output}

    def load_weights(self, filename):
        state_dict = torch.load(filename, map_location=torch.device(self.device))
        self.model.load_state_dict(state_dict, strict=False)

    def get_training_metrics(self):
        print('Training Metrics:')
        print(f'Rewards: {self.reward_history[-10:]}')
        print(f'Baseline: {self.baseline_history[-10:]}')
        print(f'Loss: {self.loss_history[-10:]}')
