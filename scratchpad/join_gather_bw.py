"""Join trace AllGather kernels to HLO all-gather output sizes by hlo_op id, compute
exact effective bandwidth per kernel and aggregate, plus a breakdown by the scope
(jit op_name prefix) that issued the gather."""
import gzip, json, re, sys
from collections import defaultdict

DTYPE_BYTES = {"f32":4,"bf16":2,"f16":2,"s32":4,"s8":1,"f8e4m3fn":1,"f8e5m2":1,"pred":1,"u32":4}
SHAPE_RE = re.compile(r"(f32|bf16|f16|s32|s8|f8e4m3fn|f8e5m2|pred|u32)\[([0-9,]*)\]")
MESH_RE = re.compile(r"mesh\[([^\]]*)\]")

def elems(d):
    if d=="" : return 1
    n=1
    for x in d.split(","): n*=int(x)
    return n

def parse_mesh(line):
    m=MESH_RE.search(line)
    out={}
    if m:
        for p in m.group(1).split(","):
            kv=p.split("=")
            if len(kv)==2: out[kv[0].strip().strip("'")]=int(kv[1])
    return out

def axes_used(line):
    cands=re.findall(r"\{([^}]*axis_[^}]*)\}", line)
    if not cands: return []
    return [a.strip().strip("'") for a in cands[-1].split(",") if "axis" in a]

def grp(line):
    mesh=parse_mesh(line); used=axes_used(line)
    g=1
    for a in used: g*=mesh.get(a,1)
    return g if used else 0

def out_bytes(line):
    # line: %all-gather-start.N = <RESULT_SHAPE> all-gather-start(<args>)
    # RESULT_SHAPE is (input, output) for single, or ((inputs),(outputs)) for tuples.
    # Take the text between the first '=' and ' all-gather-start(' = the result shape,
    # then take the OUTPUT half: for a tuple-of-tuples the 2nd top-level group; for the
    # simple 2-tuple the 2nd element. Robust heuristic: the gathered (larger) shapes are
    # the OUTPUT; sum the second half of the shape list.
    m=re.search(r"=\s*(.*?)\s*all-gather-start\(", line)
    if not m: return 0
    res=m.group(1)
    shapes=[(dt,d) for dt,d in SHAPE_RE.findall(res)]
    if not shapes: return 0
    half=len(shapes)//2
    out=shapes[half:] if half>0 else shapes
    return sum(elems(d)*DTYPE_BYTES[dt] for dt,d in out)

def load_hlo(path):
    # map "all-gather-start.<id>" -> (out_bytes, group_size)
    info={}
    with open(path) as f:
        for line in f:
            if "all-gather-start" in line and "all-gather-start(" in line:
                m=re.search(r"%(all-gather-start[.\d]*)\s*=", line)
                if m:
                    info[m.group(1)]=(out_bytes(line), grp(line))
    return info

def main(trace, hlo):
    info=load_hlo(hlo)
    print(f"HLO all-gather-start ops: {len(info)}")
    data=json.load(gzip.open(trace,'rt'))
    ev=data["traceEvents"]
    pid_name={e["pid"]:e["args"]["name"] for e in ev if e.get("ph")=="M" and e.get("name")=="process_name"}
    gpu0=[p for p,n in pid_name.items() if n=="/device:GPU:0"][0]

    # by group size: total bytes moved (per-rank received = out*(g-1)/g), total time
    by_g=defaultdict(lambda:[0,0.0,0.0])   # g -> count, total_us, total_perrank_bytes
    by_scope=defaultdict(lambda:[0,0.0,0.0])
    matched=0; unmatched=0; unmatched_us=0.0
    bw_samples=[]
    for e in ev:
        if e.get("ph")!="X" or e.get("pid")!=gpu0: continue
        if "AllGather_RING_LL" not in e.get("name",""): continue
        a=e.get("args",{})
        hop=a.get("hlo_op","")
        dur=e["dur"]
        scope=a.get("name","").replace("jit(step)/","")
        scope=re.split(r"/(while|dot_general|jvp|closed_call|body|cond)", scope)[0]
        if hop in info:
            ob,g=info[hop]
            perrank = ob*(g-1)/g if g>0 else ob
            by_g[g][0]+=1; by_g[g][1]+=dur; by_g[g][2]+=perrank
            by_scope[scope][0]+=1; by_scope[scope][1]+=dur; by_scope[scope][2]+=perrank
            matched+=1
            if dur>10: bw_samples.append((perrank/1e6)/(dur/1e6))  # GB/s alg? perrank GB / s
        else:
            unmatched+=1; unmatched_us+=dur
    print(f"matched kernels: {matched}  unmatched: {unmatched} ({unmatched_us/1e3:.1f} ms)")
    print()
    print(f"{'group':>6s} {'count':>6s} {'time_ms':>9s} {'perrank_GB':>11s} {'busBW_GB/s':>11s}")
    for g in sorted(by_g):
        c,t,b=by_g[g]
        bw = (b/1e9)/(t/1e6) if t>0 else 0
        print(f"{g:6d} {c:6d} {t/1e3:9.1f} {b/1e9:11.2f} {bw:11.0f}")
    print()
    print("=== by issuing scope (op_name prefix) ===")
    print(f"{'scope':40s} {'count':>6s} {'time_ms':>9s} {'perrank_GB':>10s} {'busBW':>7s}")
    for s in sorted(by_scope, key=lambda x:-by_scope[x][1]):
        c,t,b=by_scope[s]
        bw=(b/1e9)/(t/1e6) if t>0 else 0
        print(f"{s[:40]:40s} {c:6d} {t/1e3:9.1f} {b/1e9:10.2f} {bw:7.0f}")

if __name__=="__main__":
    main(sys.argv[1], sys.argv[2])
