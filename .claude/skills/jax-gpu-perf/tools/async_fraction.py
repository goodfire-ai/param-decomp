"""Fraction of gather-transfer time on the COMPUTE stream (serialized) vs a separate async
stream, and genuine concurrency. Usage: python async_fraction.py <xplane.pb>"""
import sys, re, collections
import google.protobuf.runtime_version as _rv
_rv.ValidateProtobufRuntimeVersion=lambda *a,**k:None
from xplane_pb2 import XSpace
COLL=re.compile(r"all-?gather|reduce-?scatter|all-?reduce|nccl",re.I)
MEMCPY=re.compile(r"memcpy|memset|copy-start|copy-done",re.I)
def kind(n): return "gather" if COLL.search(n) else ("memcpy" if MEMCPY.search(n) else "compute")
xs=XSpace(); xs.ParseFromString(open(sys.argv[1],"rb").read())
gpu=[p for p in xs.planes if "/device:GPU" in p.name][0]; em=gpu.event_metadata
compute_stream=None; best=0; streams={}
for line in gpu.lines:
    iv=[]; gb=0; cb=0
    for ev in line.events:
        d=ev.duration_ps/1e3
        if d<=0: continue
        k=kind(em[ev.metadata_id].name); s=line.timestamp_ns+ev.offset_ps/1e3
        iv.append((s,s+d,k))
        if k=="gather": gb+=d
        elif k=="compute": cb+=d
    if iv: streams[line.name]=(iv,gb,cb)
    if cb>best: best=cb; compute_stream=line.name
gb_on_compute=streams[compute_stream][1] if compute_stream else 0
gb_total=sum(g for _,g,_ in streams.values())
# concurrency: gather intervals vs compute intervals (any stream)
G=[iv for ivs,_,_ in streams.values() for iv in ivs if iv[2]=="gather"]
C=[iv for ivs,_,_ in streams.values() for iv in ivs if iv[2]=="compute"]
pts=[(s,1,0) for s,_,_ in G]+[(e,-1,0) for _,e,_ in G]+[(s,1,1) for s,_,_ in C]+[(e,-1,1) for _,e,_ in C]
pts.sort(); a=b=0; last=None; both=0.0
for t,d,w in pts:
    if last is not None and t>last and a>0 and b>0: both+=t-last
    a+=d if w==0 else 0; b+=d if w==1 else 0; last=t
print(f"gather total={gb_total/1e6:.0f}ms | on COMPUTE stream(serialized)={gb_on_compute/1e6:.0f}ms ({100*gb_on_compute/max(gb_total,1):.0f}%) | concurrency(both)={both/1e6:.0f}ms ({100*both/max(gb_total,1):.0f}%)")
