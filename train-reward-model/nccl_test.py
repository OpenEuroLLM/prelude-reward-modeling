#!/usr/bin/env python3
"""All-gather / reduce-scatter bandwidth across the job's ranks (to check that RCCL uses the Slingshot plugin)."""
import os, time, datetime
import torch, torch.distributed as dist
rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dev = torch.device("cuda", local)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev, timeout=datetime.timedelta(minutes=10))
for mb in (64, 512):
    n = mb * 2**20 // 2
    x = torch.randn(n // world * world, dtype=torch.bfloat16, device=dev)
    out = torch.empty_like(x)
    shard = x[: n // world]
    for name, fn in (("all_gather", lambda: dist.all_gather_into_tensor(out, shard)),
                     ("reduce_scatter", lambda: dist.reduce_scatter_tensor(shard, x)),
                     ("all_reduce", lambda: dist.all_reduce(x))):
        fn(); torch.cuda.synchronize(); dist.barrier()
        t = time.time()
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        dt = (time.time() - t) / 5
        if rank == 0:
            print("{} {} MB x {} ranks: {:.3f}s, {:.1f} GB/s algbw".format(name, mb, world, dt, mb / 1024 / dt), flush=True)
dist.destroy_process_group()
