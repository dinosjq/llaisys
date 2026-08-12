#!/usr/bin/env python3
"""v9 vs v6 vs FlashInfer benchmark."""
import torch, sys, os, ctypes, csv, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
NH,NKVH,HD,TN=12,2,128,64; DEV=torch.device("cuda:0"); BV2=2
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV6.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*5
lib.llaisysFlashDecodingV9.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3
fi_ws = torch.empty(128*1024*1024, dtype=torch.float32, device="cuda")

configs = [(1,256),(1,2048),(1,4096),(4,512),(4,2048),(8,1024),(16,512),(32,256)]
script = os.path.join(os.path.dirname(__file__), '_run_v9.py')
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
lib.llaisysFlashDecodingV9.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3
NH,NKVH,HD,TN=12,2,128,64; DEV=torch.device('cuda:0'); BV2=2
configs = {repr(configs)}
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
    av=torch.empty((batch,NH,HD),dtype=torch.bfloat16,device=DEV)
    acc=torch.zeros((tb,NH,HD),dtype=torch.float32,device=DEV)
    asum=torch.zeros((tb,NH,1),dtype=torch.float32,device=DEV)
    amax=torch.zeros((tb,NH,1),dtype=torch.float32,device=DEV)
    # v6 H6T16C1S2U1
    a6=(av.data_ptr(),acc.data_ptr(),asum.data_ptr(),amax.data_ptr(),
        qt.data_ptr(),kt.data_ptr(),vt.data_ptr(),tbids.data_ptr(),cut.data_ptr(),tot.data_ptr(),
        TN,batch,mb,tb,1,BV2,s,NH,HD,HD,NKVH,6,16,1,2,1)
    for _ in range(3): lib.llaisysFlashDecodingV6(*a6)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"V6_B{{batch}}T{{totlen}}")
    lib.llaisysFlashDecodingV6(*a6); torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    # v9 H6T8C1 (or H6T16)
    a9=(av.data_ptr(),acc.data_ptr(),asum.data_ptr(),amax.data_ptr(),
        qt.data_ptr(),kt.data_ptr(),vt.data_ptr(),tbids.data_ptr(),cut.data_ptr(),tot.data_ptr(),
        TN,batch,mb,tb,1,BV2,s,NH,HD,HD,NKVH,6,8,1)
    for _ in range(3): lib.llaisysFlashDecodingV9(*a9)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"V9_B{{batch}}T{{totlen}}")
    lib.llaisysFlashDecodingV9(*a9); torch.cuda.synchronize()
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
subprocess.run(["nsys","profile","-t","cuda,nvtx","-o","/tmp/v9b","--force-overwrite=true",
                "python3",script], check=False, timeout=900, capture_output=True, text=True)
subprocess.run(["rm","-f","/tmp/v9b_t_cuda_gpu_trace.csv","/tmp/v9b.sqlite"], check=False)
subprocess.run(["nsys","stats","--report=cuda_gpu_trace","--format=csv","--force-export=true",
                "-o","/tmp/v9b_t","/tmp/v9b.nsys-rep"], check=True, timeout=120, capture_output=True)

kernels = []
with open("/tmp/v9b_t_cuda_gpu_trace.csv") as f:
    r = csv.reader(f); h = next(r); ni = h.index("Name"); di = h.index("Duration (ns)")
    for row in r:
        n = row[ni]; d = int(row[di])
        if 'flash_decoding_v6' in n or 'flash_decoding_v9' in n or 'BatchDecodeWithPagedKVCacheKernel' in n:
            kernels.append((d, n))
print(f"Total: {len(kernels)}")
per = 20  # v6 8 (3w+1m x2k) + v9 8 (3w+1m x2k) + FI 4 (3w+1m x1k)
print(f"\n{'b×totlen':>14} {'v6(us)':>8} {'v9(us)':>8} {'FI(us)':>8} {'v9/v6':>7} {'FI/v9':>7}")
for ci,(b,t) in enumerate(configs):
    base=ci*20
    v6_t=kernels[base+6][0]/1e3+kernels[base+7][0]/1e3
    v9_t=kernels[base+14][0]/1e3+kernels[base+15][0]/1e3
    fi_t=kernels[base+19][0]/1e3
    print(f"  {b:2d}×{t:5d}  {v6_t:6.1f}  {v9_t:6.1f}  {fi_t:6.1f}  {v9_t/v6_t:.2f}x  {fi_t/v9_t:.2f}x")
