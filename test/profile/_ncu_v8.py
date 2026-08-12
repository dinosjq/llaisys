import torch, sys, os, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys
NH,NKVH,HD,TN=12,2,128,64; DEV=torch.device("cuda:0"); BV2=2
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV8.argtypes = [c_void_p]*6 + [c_size_t]*4 + [c_int, c_float] + [c_size_t]*4
batch,totlen=16,512
bn=totlen//TN; mb=bn+1; tb=batch*bn
tbids=torch.zeros((batch,mb),dtype=torch.int64,device=DEV)
for b in range(batch):
    tbids[b,0]=(b+1)*bn; tbids[b,1:]=torch.arange(b*bn,(b+1)*bn,device=DEV)
qt=torch.randn((batch,NH,HD),dtype=torch.bfloat16,device=DEV)
kt=torch.randn((tb,TN,NKVH,HD),dtype=torch.bfloat16,device=DEV)
vt=torch.randn((tb,TN,NKVH,HD),dtype=torch.bfloat16,device=DEV)
tot=torch.full((batch,),totlen,dtype=torch.int64,device=DEV)
s=1.0/(HD**0.5)
av=torch.empty((batch,NH,HD),dtype=torch.bfloat16,device=DEV)
args=(av.data_ptr(),qt.data_ptr(),kt.data_ptr(),vt.data_ptr(),tbids.data_ptr(),tot.data_ptr(),
      TN,batch,mb,1,BV2,s,NH,HD,HD,NKVH)
for _ in range(5): lib.llaisysFlashDecodingV8(*args)
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("V8")
lib.llaisysFlashDecodingV8(*args); torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
print("done")
