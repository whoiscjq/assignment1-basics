import torch
from tests.adapters import run_train_bpe

def save_checkpoint(model, optimizer,iteration,out):
    checkpoint={"model":model.state_dict(),"optimizer":optimizer.state_dict, "iteration":iteration}
    torch.save(checkpoint,out)

def load_checkpoint(src, model, optimizer):
    checkpoint=torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]

CONTEXT_LENGTH=2048

run_get_batch(dataset: npt.NDArray, batch_size: int, context_length=CONTEXT_LENGTH, device="cuda:0") 