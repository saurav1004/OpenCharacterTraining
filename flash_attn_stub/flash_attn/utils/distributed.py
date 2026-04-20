import torch
import torch.distributed as dist


def all_gather(tensor, group=None):
    if not dist.is_available() or not dist.is_initialized():
        return [tensor]
    world_size = dist.get_world_size(group=group)
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=group)
    return gathered
