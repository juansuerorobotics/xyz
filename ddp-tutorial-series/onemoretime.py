import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

import torch.distributed as dist

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int, 
    ) -> None:
        self.gpu_id = gpu_id
        
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        
        #self.model = model.to(gpu_id)
        self.model = DDP(model.to(gpu_id), device_ids=[gpu_id])
        
        

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        #loss = F.cross_entropy(output, targets)
        loss = F.mse_loss(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        #ckp = self.model.state_dict()
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            # IMPORTANT: shuffle shards differently each epoch
            if isinstance(self.train_data.sampler, DistributedSampler):
                self.train_data.sampler.set_epoch(epoch)

            self._run_epoch(epoch)

            # only rank 0 saves
            if dist.get_rank() == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    
    # use global world size & rank
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True, 
        sampler=sampler
    )

import os
def cleanup_ddp():
    dist.destroy_process_group()
    
    
def setup_ddp():
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    
    
    print(
        f"[ddp_min] starting on host={socket.gethostname()} "
        f"rank={rank}/{world_size} "
        f"MASTER={master_addr}:{master_port}",
        flush=True,
    )

    print(f"[ddp_min] rank={rank} calling init_process_group()", flush=True)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )

    print(f"[ddp_min] rank={rank} init_process_group() returned", flush=True)

    # simple sync to prove everything works
    dist.barrier()
    print(f"[ddp_min] rank={rank} passed barrier()", flush=True)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[setup_ddp] rank={rank} world_size={world_size}", flush=True)

    return rank, world_size

def main(args):
    rank, world_size = setup_ddp()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    gpu_id = local_rank
    print(f"[Rank {rank}] Starting on {socket.gethostname()} using GPU {gpu_id}", flush=True)



    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, args.batch_size)
    trainer = Trainer(model, train_data, optimizer, gpu_id, args.save_every)
    trainer.train(args.total_epochs)
    
    # ----------------------------------
    # Evaluation (Rank 0 only)
    # ----------------------------------
    if dist.get_rank() == 0:
        trainer.model.eval()
        
        
        with torch.no_grad():  # no gradients while testing
            
            total_loss = 0
            steps = 0
            for source, targets in trainer.train_data:
                source = source.to(trainer.gpu_id)    # [batch_size, 20]
                targets = targets.to(trainer.gpu_id)  # [batch_size]
                output = trainer.model(source)        # [batch_size, 1] in your case
                
                #loss = F.cross_entropy(output, targets)
                loss = F.mse_loss(output, targets)
                
                total_loss += loss.item()
                steps += 1   
            avg_loss = total_loss / steps
            print(f"Test avg loss: {avg_loss:.4f}") 
        
        
    cleanup_ddp()
    
import socket
if __name__ == "__main__":
    import argparse
    
    #import torch.multiprocessing as mp   
    
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    
    args = parser.parse_args()
    main(args)

