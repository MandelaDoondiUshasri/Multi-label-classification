from dataset.loader import ODIR5K
from torch.utils.data import DataLoader
from train import TrainManager
import numpy as np
import torch
import random
import os
from loguru import logger as printer
import argparse
class Trainer:

    def __init__(self, params):
        self.params = params
        self.set_seed(42)

        if self.params["logging_active"]:
            printer.add(os.path.join(self.params["log_dir"],self.params["exp_name"])+"/{time}.log")

        train_dataset = ODIR5K(self.params.get("img_dir"),
                                     self.params.get("label_dir"),
                                     train_test_size=self.params.get("train_test_size"), 
                                     is_train=True, augment=self.params.get("augment") )
        
        test_dataset = ODIR5K(self.params.get("img_dir"),
                                     self.params.get("label_dir"),
                                     train_test_size=self.params.get("train_test_size"), 
                                     is_train=False, augment=self.params.get("augment"))
        
        self.train_loader = DataLoader(dataset=train_dataset,
                              batch_size=self.params["batch_size"],
                              shuffle=self.params["shuffle"],
                              num_workers=self.params["num_workers"])
        self.test_loader = DataLoader(dataset=test_dataset,
                              batch_size=self.params["batch_size"],
                              shuffle=self.params["shuffle"],
                              num_workers=self.params["num_workers"])
        
    def set_seed(self, seed: int = 42) -> None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        printer.info(f"Random seed set as {seed}")


    def run(self):
        train_manager = TrainManager(train_loader=self.train_loader,
                                     test_loader=self.test_loader, 
                                     **self.params)
                                    
        train_manager.train()
        



    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--exp_name', default="Default")
    parser.add_argument('--img_dir', default="/kaggle/input/ocular-disease-recognition-odir5k/preprocessed_images")
    parser.add_argument('--label_dir', default=/kaggle/input/dataset/full_df.csv")
    parser.add_argument('--log_dir', default="./docs/logs")
    parser.add_argument('--logger_name', default="tensorboard")
    parser.add_argument('--logging_active', default=True, type=bool)
    parser.add_argument('--vis_print_per_iter', default=5, type=int)
    parser.add_argument('--test_per_iter', default=25, type=int)
    parser.add_argument('--model_name', default="EfficientNetB3Pretrained")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--train_test_size', default=0.9, type=float)
    parser.add_argument('--device', default="cpu", type=str, choices=["cpu", "cuda"])
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--augment', default=None)
    parser.add_argument('--load_model', default=False, type=bool)
    parser.add_argument('--load_model_path', default="./docs/logs/Mod.pth")
    args = parser.parse_args()
    params = vars(args)
    print(params)
    
    trainer = Trainer(params)
    trainer.run()
