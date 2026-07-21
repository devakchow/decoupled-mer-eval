import glob, os, json, collections
import pretty_midi
L="/scratch/gilbreth/dcharapa/mer/run/data/CocoChorales-E/label"
R="/scratch/gilbreth/dcharapa/mer/run"
cls=["extra_notes","removed_notes","correct_notes"]
def nn(f):
    pm=pretty_midi.PrettyMIDI(f); return sum(len(i.notes) for i in pm.instruments)
gt={}
for c in cls:
    for f in glob.glob(f"{L}/{c}/*/stems_midi/*.mid"):
        tid=f.split("/label/")[1].split("/")[1]
        stem=os.path.splitext(os.path.basename(f))[0]   # e.g. 0_flute
        gt.setdefault(f"{tid}_{stem}",{})[c]=nn(f)
gtpat={k:tuple(1 if v.get(c,0)>0 else 0 for c in cls) for k,v in gt.items()}
print("gt stems",len(gtpat),flush=True)
res={}
for st in ["C_polytune_coco","D_laddersym_coco_prompted","D_laddersym_coco_unprompted"]:
    joint=collections.Counter(); miss=0
    for f in sorted(glob.glob(f"{R}/staging/{st}/*.mid")):
        key=os.path.splitext(os.path.basename(f))[0]
        pm=pretty_midi.PrettyMIDI(f); n=sum(1 for i in pm.instruments if i.notes)
        p=gtpat.get(key)
        if p is None: miss+=1; continue
        joint[(n,p)]+=1
    # front-pad ceiling: pad correct requires GT present-set == suffix of (e,r,c) of size n
    def suffix(n): return tuple([0]*(3-n)+[1]*n)
    ceil=sum(v for (n,p),v in joint.items() if p==suffix(n))
    tot=sum(joint.values())
    res[st]={"unmatched_keys":miss,"n":tot,
      "joint":{f"npred={n} gt={p}":v for (n,p),v in sorted(joint.items())},
      "frontpad_ceiling":ceil,"frontpad_ceiling_frac":ceil/tot if tot else None}
    # restricted to under-specified preds
    u={k:v for k,v in joint.items() if k[0]<3}
    ut=sum(u.values()); uc=sum(v for (n,p),v in u.items() if p==suffix(n))
    res[st]["under_n"]=ut; res[st]["under_frontpad_ceiling"]=uc
    res[st]["under_frontpad_ceiling_frac"]=uc/ut if ut else None
    print(st,json.dumps(res[st],indent=1),flush=True)
json.dump(res,open(f"{R}/results/VERIFY_joingt.json","w"),indent=1)
