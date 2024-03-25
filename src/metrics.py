import os
import wandb
import numpy as np


class Metrics:
    def __init__(self, training_class):
        self.configs = training_class.configs
        self.initialize_writer()
        self.delete_file('../wandb/debug.log')
        self.delete_file('../wandb/debug-internal.log')
    

    def initialize_writer(self):
        wandb.init(
            project='S5_pendulum_mine',
            name=self.configs['run_name'],
            dir='../',
            config=self.configs
        )

    
    def delete_file(self, file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"{file_path} has been deleted.")
        else:
            print(f"The file {file_path} does not exist.")

    
    def add_train_metrics(self, loss, lr, step):
        wandb.log({"Train Loss": loss}, step=step)
        wandb.log({"Learning Rate": lr}, step=step)

    
    def add_val_loss(self, loss, step):
        for _ in range(11):
            wandb.log({f"Val Loss x_{10*_}": loss[10*_,0]}, step=step)
            wandb.log({f"Val Loss y_{10*_}": loss[10*_,1]}, step=step)
        wandb.log({"Val Loss": np.mean(loss)}, step=step)
        

    def close_writer(self):
        wandb.finish()

