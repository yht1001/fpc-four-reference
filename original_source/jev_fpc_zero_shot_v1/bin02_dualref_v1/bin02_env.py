"""Shared explicit inputs; no SFT weights, no Query labels in requests."""
import datetime, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent
FPC = R.parent
PROJECT = FPC.parent
MODEL = PROJECT/'StartLux-Decision/models/StartLux-Decision-v1.1-4B-AutoThink'
PY = PROJECT/'flora_jev_fpc_v1/.venv/bin/python'
VLLM_PY = PROJECT/'StartLux-Decision/.venv-vllm031/bin/python'
PREVIOUS = PROJECT/'industryasm_sft_transfer_v1/temporal_hardbg_v1'
OLD = FPC/'probability_alignment_v1/reference_ensemble'
FRAMES = PROJECT/'datasets/FPCdataset2601/frames/ffpmeg384fps25'
GT = PROJECT/'datasets/FPCdataset2601/annotation/gtTXT'
REFS = {'original':FPC/'support/references', 'diverse':FPC/'probability_alignment_v1/multivideo_references/references'}
LABELS = ['A01','A02','A03','A04','A05','BG']
sys.path.insert(0,str(FPC/'scripts'))
sys.path.insert(0,str(PROJECT/'StartLux-Decision'))

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def fingerprint(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n');tmp.replace(p)
def jsonl(p): return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
def check(lock):
    for p,h in lock['source_sha256'].items():assert sha(p)==h,p
def rows(split):return read(R/'inputs'/f'{split}.json')['windows']
def request(row,group):
    from run_query import build_request
    req=build_request(row,{},'never','visual_current',REFS[group])
    if row.get('target_duration_s')==.2:
        req['state']=req['state'].replace('0.5 seconds','0.2 seconds')
        req['questions']['action']['instructions']=req['questions']['action']['instructions'].replace('0.5 seconds','0.2 seconds')
    return req
def signature(row,group):
    from run_query import build_request
    req=build_request({'sheet':'/dev/null'}, {},'never','visual_current',REFS[group])
    if row.get('target_duration_s')==.2:
        req['state']=req['state'].replace('0.5 seconds','0.2 seconds')
        req['questions']['action']['instructions']=req['questions']['action']['instructions'].replace('0.5 seconds','0.2 seconds')
    del req['images']
    return fingerprint({'request_without_image_data':req,'images_sha256':[row['sheet_sha256']]+read(R/'REFERENCES.json')[group]})
def inputs(split):
    return [dict(r,id=f'{split}:{r["video_id"]}:{r["window_id"]}:{g}',reference_group=g,input_sha256=r['request_sha256'][g]) for r in rows(split) for g in REFS]
def render(row,tokenizer):
    from startlux_decide.prompt import from_systemone,messages,plain_prompt
    from startlux_decide.images import place_images
    from PIL import Image
    req=request(row,row['reference_group'])
    prompt=plain_prompt(tokenizer,messages(from_systemone(place_images(req['state'],7),req['questions']['action'])))
    images=[]
    for p in [Path(row['sheet'])]+[REFS[row['reference_group']]/(l+'.jpg') for l in LABELS]:
        with Image.open(p) as im:images.append(im.convert('RGB'))
    return prompt,images
def outputs(split):
    result={}
    for p in sorted((R/'raw').glob(f'{split}_rank*.jsonl')):
        for r in jsonl(p):
            assert r['id'] not in result,r['id'];result[r['id']]=r
    return result
def aggregate(split):
    result=outputs(split);expected=inputs(split)
    assert set(result)=={r['id'] for r in expected},(split,len(result),len(expected))
    for r in expected:
        out=result[r['id']];p=np.array(out['probabilities'])
        assert out['input_sha256']==r['input_sha256'] and p.shape==(6,) and np.isfinite(p).all() and (p>=0).all() and abs(p.sum()-1)<1e-6
    return np.array([np.mean([result[f'{split}:{r["video_id"]}:{r["window_id"]}:{g}']['probabilities'] for g in REFS],axis=0) for r in rows(split)])
def truth(rows_):
    from evaluate_query import GT_TO_OPTION
    table={v:[GT_TO_OPTION[x.strip()] for x in (GT/f'{v}.txt').read_text().splitlines()] for v in {r['video_id'] for r in rows_}}
    return np.array([LABELS.index(table[r['video_id']][int(r['target_center_s']*25+1e-7)]) for r in rows_])
def decode(rows_,p,threshold,scale=.4):
    import evaluate_ensemble_temporal_prior as et
    import evaluate_soft_temporal_prior as st
    result=np.full(len(rows_),-1,dtype=int)
    for v in sorted({r['video_id'] for r in rows_}):
        idx=[i for i,r in enumerate(rows_) if r['video_id']==v];idx.sort(key=lambda i:rows_[i]['window_id'])
        result[idx]=st.viterbi(et.threshold_emissions(np.array(threshold),p[idx])*scale,0,.75)
    assert (result>=0).all()
    return result
