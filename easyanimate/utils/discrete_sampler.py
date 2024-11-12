"""Modified from https://github.com/THUDM/CogVideo/blob/3710a612d8760f5cdb1741befeebb65b9e0f2fe0/sat/sgm/modules/diffusionmodules/sigma_sampling.py
"""
import torch

class DiscreteSampling:
    def __init__(self, num_idx, uniform_sampling=False):
        self.num_idx = num_idx
        self.uniform_sampling = uniform_sampling
        self.is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

        if self.is_distributed and self.uniform_sampling:
            world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

            i = 1
            while True:
                if world_size % i != 0 or num_idx % (world_size // i) != 0:
                    i += 1
                else: 
                    self.group_num = world_size // i
                    break
            assert self.group_num > 0 
            assert world_size % self.group_num == 0 
            # the number of rank in one group 
            self.group_width = world_size // self.group_num  
            self.sigma_interval = self.num_idx // self.group_num
            print('rank=%d world_size=%d group_num=%d group_width=%d sigma_interval=%s' % (
                  self.rank, world_size, self.group_num,
                  self.group_width, self.sigma_interval))
        
    def __call__(self, n_samples, generator=None, device=None):
        if self.is_distributed and self.uniform_sampling: 
            group_index = self.rank // self.group_width
            idx = torch.randint(
                    group_index * self.sigma_interval,
                    (group_index + 1) * self.sigma_interval,
                    (n_samples,), 
                    generator=generator, device=device,
                )
            print('proc[%d] idx=%s' % (self.rank, idx))
        else:   
            idx = torch.randint(
                    0, self.num_idx, (n_samples,), 
                    generator=generator, device=device,
                )
        return idx