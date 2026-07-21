import glob, os, json, collections
import pretty_midi
L = "/scratch/gilbreth/dcharapa/mer/run/data/CocoChorales-E/label"
cls = ["extra_notes","removed_notes","correct_notes"]
def nn(f):
    try:
        pm = pretty_midi.PrettyMIDI(f)
        return sum(len(i.notes) for i in pm.instruments)
    except Exception as e:
        return -1
stems = {}
for c in cls:
    for f in glob.glob(f"{L}/{c}/*/stems_midi/*.mid"):
        tid = f.split("/label/")[1].split("/")[1]
        key = (tid, os.path.basename(f))
        stems.setdefault(key, {})[c] = nn(f)
print("n stems", len(stems), flush=True)
pat = collections.Counter()
emptycnt = collections.Counter()
for k,v in stems.items():
    p = tuple(1 if v.get(c,0)>0 else 0 for c in cls)  # present flags
    pat[p]+=1
    for i,c in enumerate(cls):
        if p[i]==0: emptycnt[c]+=1
tot=len(stems)
res = {"n_stems": tot,
       "present_pattern_counts": {str(k):v for k,v in sorted(pat.items(), key=lambda x:-x[1])},
       "empty_per_class": dict(emptycnt),
       "empty_frac": {c: emptycnt[c]/tot for c in cls}}
# front-pad correctness: correct iff absent classes are a prefix of (extra,removed,correct)
def prefix_ok(p):
    absent=[i for i,x in enumerate(p) if x==0]
    return absent == list(range(len(absent)))
ok=sum(v for k,v in pat.items() if prefix_ok(k))
res["gt_prefix_compatible"]=ok
res["gt_prefix_frac"]=ok/tot
json.dump(res, open("/scratch/gilbreth/dcharapa/mer/run/results/VERIFY_gtcensus.json","w"), indent=1)
print(json.dumps(res, indent=1))
