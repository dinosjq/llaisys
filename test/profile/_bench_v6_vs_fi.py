#!/usr/bin/env python3
"""v6 optimal vs FlashInfer NH=12 (patched group_size=6) — 同环境 nsys 实测."""
import torch, sys, os, ctypes, csv, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

NH, NKVH, HD, TN = 12, 2, 128, 64
DEV = torch.device("cuda:0"); BV2 = 2
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV6.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*5
fi_ws = torch.empty(128*1024*1024, dtype=torch.float32, device="cuda")

configs = [(1,256),(1,1024),(1,2048),(1,4096),(4,512),(4,2048),(8,1024),(16,512),(32,256)]
# v6 optimal per config (from sweep)
best_params = {
    (1,256): (1,16,1,2,0), (1,1024): (2,16,1,2,1), (1,2048): (6,16,1,2,1),
    (1,4096): (6,16,1,2,1), (4,512): (6,16,1,2,1), (4,2048): (6,16,1,2,1),
    (8,1024): (6,16,1,2,1), (16,512): (6,16,1,2,1), (32,256): (6,16,1,2,1),
}

script = os.path.join(os.path.dirname(__file__), '_run_v6fi.py')
with open(script,'w') as f:
    f.write(f"""#!/usr/bin/env python3
import torch, sys, os, ctypes
sys.path.insert(0, r'{os.path.join(os.path.dirname(__file__), '..')}')
sys.path.insert(0, r'{os.path.join(os.path.dirname(__file__), '..', '..', 'python')}')
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV6.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*5
NH,NKVH,HD,TN=12,2,128,64; DEV=torch.device('cuda:0'); BV2=2
configs = {repr(configs)}; best_params = {repr(best_params)}
fi_ws = torch.empty(128*1024*1024, dtype=torch.float32, device='cuda')
for (batch, totlen) in configs:
    bn=totlen//TN; mb=bn+1; tb=batch*bn
    tbids=torch.zeros((batch,mb),dtype=torch.int64,device=DEV)
    for b in range(batch):
        tbids[b,0]=(b+1)*bn; tbids[b,1:]=torch.arange(b*bn,(b+1)*bn,device=DEV)
    qt=torch.randn((batch,NH,HD),dtype=torch.bfloat16,device=DEV)
    kt=torch.randn((tb,TN,NKVH,HD),dtype=torch.bfloat16,device=DEV)
    vt=torch.randn((tb,TN,NKVH,HD),dtype=torch.bfloat16,device=DEV)
    cut=torch.arange(batch+1,dtype=torch.int64,device=DEV)
    tot=torch.full((batch,),totlen,dtype=torch.int64,device=DEV)
    s=1.0/(HD**0.5)
    # v6
    h,tk,co,ns,ur = best_params[(batch,totlen)]
    av=torch.empty((batch,NH,HD),dtype=torch.bfloat16,device=DEV)
    acc=torch.zeros((tb,NH,HD),dtype=torch.float32,device=DEV)
    asum=torch.zeros((tb,NH,1),dtype=torch.float32,device=DEV)
    amax=torch.zeros((tb,NH,1),dtype=torch.float32,device=DEV)
    args=(av.data_ptr(),acc.data_ptr(),asum.data_ptr(),amax.data_ptr(),
          qt.data_ptr(),kt.data_ptr(),vt.data_ptr(),tbids.data_ptr(),cut.data_ptr(),tot.data_ptr(),
          TN,batch,mb,tb,1,BV2,s,NH,HD,HD,NKVH,h,tk,co,ns,ur)
    for _ in range(3): lib.llaisysFlashDecodingV6(*args)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"V6_B{{batch}}T{{totlen}}")
    lib.llaisysFlashDecodingV6(*args); torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    # FlashInfer
    indptr = torch.arange(0, batch+1, dtype=torch.int32, device=DEV) * bn
    indices = torch.tile(torch.arange(bn, dtype=torch.int32, device=DEV), (batch,))
    last_page = torch.full((batch,), TN, dtype=torch.int32, device=DEV)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(fi_ws, kv_layout="NHD")
    wrapper.plan(indptr, indices, last_page, NH, NKVH, HD, TN, q_data_type=torch.bfloat16)
    k_fi = torch.randn(tb, TN, NKVH, HD, dtype=torch.bfloat16, device=DEV)
    v_fi = torch.randn_like(k_fi)
    for _ in range(3): wrapper.run(qt, (k_fi, v_fi))
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"FI_B{{batch}}T{{totlen}}")
    wrapper.run(qt, (k_fi, v_fi)); torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
print("done")
""")

print("Running nsys...")
subprocess.run(["nsys","profile","-t","cuda,nvtx","-o","/tmp/v6fi","--force-overwrite=true",
                "python3",script], check=False, timeout=900, capture_output=True, text=True)
subprocess.run(["rm","-f","/tmp/v6fi_t*.csv"], check=False)
subprocess.run(["nsys","stats","--report=cuda_gpu_trace","--format=csv","--force-export=true",
                "-o","/tmp/v6fi_t","/tmp/v6fi.nsys-rep"], check=True, timeout=120, capture_output=True)

# Parse: per config, v6 (par+red) then FI (1 kernel), each 3 warmup + 1 meas
# Order: v6 warmup x3 (par+red=6k), v6 meas (par+red=2k), FI warmup x3 (3k), FI meas (1k)
# Per config kernels: v6: 8 (3 warmup calls x2 + 1 meas x2), FI: 4 (3 warmup + 1 meas)
# Total per config = 12 kernels
kernels = []
with open("/tmp/v6fi_t_cuda_gpu_trace.csv") as f:
    r = csv.reader(f); h = next(r); ni = h.index("Name"); di = h.index("Duration (ns)")
    for row in r:
        n = row[ni]; d = int(row[di])
        if 'flash_decoding_v6' in n or 'BatchDecodeWithPagedKVCacheKernel' in n:
            kernels.append((d, n))

print(f"Total kernels: {len(kernels)} (expect {len(configs)*12})")
# Per config: v6 8 (3 warmup calls + meas, each 2k), FI 4 (3 warmup + meas)
per = 12
print(f"\n{'b×totlen':>14} {'v6(us)':>9} {'FlashInfer(us)':>14} {'v6/FI':>8}")
for ci, (b,t) in enumerate(configs):
    base = ci * per
    # v6 measured = last 2 of v6 block (index 6,7)
    v6_t = kernels[base+6][0]/1e3 + kernels[base+7][0]/1e3
    # FI measured = last kernel (index 11)
    fi_t = kernels[base+11][0]/1e3
    ratio = f"{fi_t/v6_t:.2f}x" if v6_t else "N/A"
    print(f"  {b:2d}×{t:5d}  {v6_t:7.1f}  {fi_t:12.1f}  {ratio}")
