import torch, sys, os, ctypes, csv, subprocess, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys
NH,NKVH,HD,TN=12,2,128,64; DEV=torch.device("cuda:0"); BV2=2
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV3.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3
lib.llaisysFlashDecodingV5.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3

configs = [(1,256),(1,512),(1,1024),(1,2048),(1,4096),(4,2048),(8,1024),(16,512),(32,256),
           (2,256),(2,2048),(4,1024),(8,512),(16,256)]
tests = [(6,8),(2,16)]  # v3 default + small-batch best

script = os.path.join(os.path.dirname(__file__), '_run_v5.py')
with open(script,'w') as f:
    f.write(f"""#!/usr/bin/env python3
import torch, sys, os, ctypes
sys.path.insert(0, r'{os.path.join(os.path.dirname(__file__), '..')}')
sys.path.insert(0, r'{os.path.join(os.path.dirname(__file__), '..', '..', 'python')}')
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV3.argtypes = lib.llaisysFlashDecodingV5.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3
NH,NKVH,HD,TN=12,2,128,64; DEV=torch.device('cuda:0'); BV2=2
configs = {repr(configs)}; tests = {repr(tests)}
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
    for (h,tk) in tests:
        for fn_name in ['llaisysFlashDecodingV3','llaisysFlashDecodingV5']:
            av=torch.empty((batch,NH,HD),dtype=torch.bfloat16,device=DEV)
            acc=torch.zeros((tb,NH,HD),dtype=torch.float32,device=DEV)
            asum=torch.zeros((tb,NH,1),dtype=torch.float32,device=DEV)
            amax=torch.zeros((tb,NH,1),dtype=torch.float32,device=DEV)
            args=(av.data_ptr(),acc.data_ptr(),asum.data_ptr(),amax.data_ptr(),
                  qt.data_ptr(),kt.data_ptr(),vt.data_ptr(),tbids.data_ptr(),cut.data_ptr(),tot.data_ptr(),
                  TN,batch,mb,tb,1,BV2,s,NH,HD,HD,NKVH,h,tk,0)
            getattr(lib,fn_name)(*args); torch.cuda.synchronize()
            label=f"{{fn_name[-2:]}}_B{{batch}}T{{totlen}}_H{{h}}K{{tk}}"
            torch.cuda.nvtx.range_push(label)
            getattr(lib,fn_name)(*args); torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
print("done")
""")

print("Running...")
r = subprocess.run(["nsys","profile","-t","cuda,nvtx","-o","/tmp/v5_bench","--force-overwrite=true","python3",script],
                   check=False, timeout=600, capture_output=True, text=True)
subprocess.run(["rm","-f","/tmp/v5_bench_t*.csv"], check=False)
subprocess.run(["nsys","stats","--report=cuda_gpu_trace","--format=csv","--force-export=true",
                "-o","/tmp/v5_bench_t","/tmp/v5_bench.nsys-rep"], check=True, timeout=120, capture_output=True)

kernels = []
with open("/tmp/v5_bench_t_cuda_gpu_trace.csv") as f:
    r = csv.reader(f); h = next(r); ni = h.index("Name"); di = h.index("Duration (ns)")
    for row in r:
        n = row[ni]
        if "flash_decoding_v3_kernel" in n or "flash_decoding_v3_reduce" in n or \
           "flash_decoding_v5_kernel" in n or "flash_decoding_v5_reduce" in n:
            kernels.append((int(row[di]), n))

# 4 kernels per combo: warmup_par, warmup_red, meas_par, meas_red
n_configs = len(configs) * len(tests) * 2  # ×2 for v3+v5
print(f"\n{'batch×totlen':>14} {'v3(us)':>9} {'v5(us)':>9} {'speedup':>8} {'H,TK':>10}")
for ci, (batch, totlen) in enumerate(configs):
    for ti, (h, tk) in enumerate(tests):
        vi = ci * len(tests) * 2 + ti * 2  # v3 index (0, 2, 4, ...)
        v3_base = vi * 4
        v5_base = (vi + 1) * 4
        v3_pu = kernels[v3_base+2][0]/1e3; v3_ru = kernels[v3_base+3][0]/1e3
        v5_pu = kernels[v5_base+2][0]/1e3; v5_ru = kernels[v5_base+3][0]/1e3
        v3_t = v3_pu+v3_ru; v5_t = v5_pu+v5_ru
        sp = f"{v3_t/v5_t:.2f}x" if v5_t > 0 else "N/A"
        print(f"  {batch:2d} × {totlen:5d}  {v3_t:8.1f} {v5_t:8.1f}  {sp:>7}  H={h} TK={tk}")
