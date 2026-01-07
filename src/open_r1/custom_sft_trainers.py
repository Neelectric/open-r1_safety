

from trl import ModelConfig, SFTTrainer
from open_r1.optims.dadamw import DAdamW, setup_dadamw

from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import get_constant_schedule_with_warmup


class SFTTrainerWithDAdamW(SFTTrainer):
    def __init__(self, *args, preconditioner_power=0.5, **kwargs):
        # Call parent __init__ with all its arguments so I pass that correctly
        super().__init__(*args, **kwargs)
        # Then I can set custom precond power
        self.preconditioner_power = preconditioner_power

    def create_optimizer(self):
        """Override to use DAdamW instead of default optimizer. A custom trainer is necessary to handle the DS Z-3 stuff correctly"""
        if self.optimizer is None:
            # Model is already wrapped by Accelerator at this point
            optimizer = setup_dadamw(self.args, self.model, self.preconditioner_power)
            self.optimizer = optimizer
        return self.optimizer
    
    def create_scheduler(self, num_training_steps, optimizer=None):
        """Override to use custom scheduler"""
        print("Currently warmup_ratio hardcoded to 0.1!")
        if optimizer is None:
            optimizer = self.optimizer
        self.lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * num_training_steps)  # warmup_ratio=0.1
        )
        return self.lr_scheduler
    
    
class SFTTrainerWithFisher(SFTTrainer):
    def __init__(self, *args, recompute_fisher_intervals=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.recompute_fisher_intervals = recompute_fisher_intervals
        
    def compute_loss(self, model, inputs):
        return