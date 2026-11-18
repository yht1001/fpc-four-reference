"""Support-only fitting followed by frozen full Query scoring."""
from bin02_env import *
import argparse
from collections import Counter
import evaluate_probability_alignment as pa
import evaluate_probability_ensemble as pe
import evaluate_soft_temporal_prior as st
from evaluate_query import GT_TO_OPTION,classification_metrics,temporal_metrics

def metrics(rows_,pred,y):
    decoded=[{'video_id':r['video_id'],'window_id':r['window_id'],'gt':LABELS[int(t)],'prediction':LABELS[int(p)]} for r,p,t in zip(rows_,pred,y)]
    m=classification_metrics(decoded);m['six_class_macro_f1']=sum(m['per_class'][l]['f1'] for l in LABELS)/6
    m['foreground_recall']=float(np.sum((y!=5)&(pred!=5))/max(np.sum(y!=5),1))
    m['action_to_BG']=int(np.sum((y!=5)&(pred==5)));m['BG_to_action']=int(np.sum((y==5)&(pred!=5)))
    m['confusion']={g:{p:0 for p in LABELS} for g in LABELS}
    for r in decoded:m['confusion'][r['gt']][r['prediction']]+=1
    per={}
    for v in sorted({r['video_id'] for r in decoded}):
        subset=[r for r in decoded if r['video_id']==v]
        per[v]={'classification':classification_metrics(subset),'temporal':{name:temporal_metrics(subset,drop) for name,drop in [('with_background',False),('without_background',True)]}}
    temporal={name:{k:float(np.mean([x['temporal'][name][k] for x in per.values()])) for k in ('Acc','Edit','F1@10','F1@25','F1@50')} for name in ('with_background','without_background')}
    for d in temporal.values():d['FEA']=sum(d.values())/5
    return {'classification':m,'temporal':temporal,'per_video':per}

def controls():
    fresh=outputs('controls');aggregate('controls')
    old={'original':pa.latest_results(pa.RAW),'diverse':pa.latest_results(pe.ALT_RAW)};diff=[]
    for r in inputs('controls'):
        a=fresh[r['id']]['probabilities'];b=old[r['reference_group']][(r['video_id'],r['window_id'])]['response']['answers']['action']['probabilities']
        diff.extend(np.abs(np.array(a)-np.array([b[l] for l in LABELS])).tolist())
    passed=float(np.mean(diff))<=.01 and max(diff)<=.08
    save(R/'RUNTIME_CONTROL.json',{'status':'passed' if passed else 'failed','requests':len(fresh),'mean_absolute_difference':float(np.mean(diff)),
        'max_absolute_difference':max(diff),'thresholds':{'mean':.01,'max':.08}})
    assert passed,'old 0.5s request drift control failed'

def select():
    check(read(R/'RUN_LOCK.json'));assert read(R/'RUNTIME_CONTROL.json')['status']=='passed'
    assert not (R/'LOCK.json').exists(),'selection already frozen'
    rs=rows('support');p=aggregate('support');y=truth(rs);groups=sorted({r['session_group'] for r in rs})
    cv=np.full(len(rs),-1,dtype=int);folds=[]
    for train in groups:
        tr=np.array([r['session_group']==train for r in rs]);va=~tr
        t=pa.fit_thresholds(p[tr],y[tr],'balanced');sub=[r for r,k in zip(rs,va) if k]
        cv[va]=decode(sub,p[va],t)
        folds.append({'train_group':train,'train_video_ids':sorted({r['video_id'] for r,k in zip(rs,tr) if k}),
            'validation_video_ids':sorted({r['video_id'] for r,k in zip(rs,va) if k}),'thresholds':t.tolist()})
        print('FIT_OUTER',train,t.tolist(),flush=True)
    t=pa.fit_thresholds(p,y,'balanced')
    save(R/'support_outer_predictions.json',[{'video_id':r['video_id'],'window_id':r['window_id'],'gt':LABELS[int(t)],'prediction':LABELS[int(a)]} for r,t,a in zip(rs,y,cv)])
    lock={'status':'locked','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':read(R/'RUN_LOCK.json')['source_sha256'],
        'support_probabilities_sha256':{str(f):sha(f) for f in sorted((R/'raw').glob('support_rank*.jsonl'))},
        'query_GT_used_for_selection':False,'SFT_used':False,'temporal':{'change_penalty':0,'order_penalty':.75,'emission_scale':.4},
        'thresholds':t.tolist(),'outer_folds':folds,'outer_support':metrics(rs,cv,y),'Query_planned_in_full':True}
    save(R/'LOCK.json',lock);print('SUPPORT_LOCKED',t.tolist(),flush=True)

def project(rs,pred,start,end):
    a=np.full(end-start,-1,dtype=int)
    for r,label in zip(rs,pred):
        lo=max(start,int(np.ceil(r['target_start_s']*25-1e-7)));hi=min(end,int(np.ceil(r['target_end_s']*25-1e-7)))
        if hi>lo:
            assert np.all(a[lo-start:hi-start]<0),'overlapping target bins';a[lo-start:hi-start]=int(label)
    assert (a>=0).all(),'uncovered comparison frames'
    return a

def old_replay():
    old=read(OLD/'temporal_prior_v1/selected_query_predictions.json');config=read(OLD/'temporal_prior_v1/evaluation.json')['fitted_on_all_support']
    p=np.array([[r['ensemble_probabilities'][l] for l in LABELS] for r in old]);t=pa.threshold_array(config['thresholds'])
    result=decode(old,p,t,scale=1)
    assert [LABELS[int(x)] for x in result]==[r['prediction'] for r in old]
    assert config['temporal']=={'change_penalty':0.,'order_penalty':.75}
    return old,result,t

def evaluate():
    lock=read(R/'LOCK.json');check(lock);check({'source_sha256':lock['support_probabilities_sha256']})
    rs=rows('query');p=aggregate('query');y=truth(rs);old,old_pred,old_t=old_replay()
    arms={'bin02_recalibrated':decode(rs,p,lock['thresholds']),'bin02_fixed_old_thresholds':decode(rs,p,old_t)}
    e={'status':'complete','lock_sha256':sha(R/'LOCK.json'),'Query_GT_used_for_selection':False,'SFT_used':False,'cached_baseline_exact_windows':len(old),
        'center_grid':{name:metrics(rs,pred,y) for name,pred in arms.items()},'common_25fps':{},
        'historical_05_center':metrics(old,old_pred,np.array([LABELS.index(r['gt']) for r in old]))}
    common_rows=[];native_y=[];native_predictions={k:[] for k in ['baseline_cached_05']+list(arms)};short={k:{'hit':0,'total':0} for k in native_predictions}
    mixed=0;coverage=read(R/'inputs/query.json')['coverage']
    for v,c in coverage.items():
        start,end=c['start_frame_zero_based'],c['end_frame_exclusive']
        gt=np.array([LABELS.index(GT_TO_OPTION[x.strip()]) for x in (GT/f'{v}.txt').read_text().splitlines()])
        yy=gt[start:end];idx=[i for i,r in enumerate(rs) if r['video_id']==v]
        oldidx=[i for i,r in enumerate(old) if r['video_id']==v]
        projected={'baseline_cached_05':project([old[i] for i in oldidx],old_pred[oldidx],start,end)}
        for name,pred in arms.items():projected[name]=project([rs[i] for i in idx],pred[idx],start,end)
        common_rows.extend({'video_id':v,'window_id':frame} for frame in range(start,end));native_y.extend(yy.tolist())
        for name,arr in projected.items():native_predictions[name].extend(arr.tolist())
        for i in idx:
            r=rs[i];mixed+=len(set(gt[r['start_frame_zero_based']:r['end_frame_exclusive']]))>1
        # Complete GT segments only; do not count a cropped fragment as a short action.
        boundaries=np.r_[0,np.flatnonzero(gt[1:]!=gt[:-1])+1,len(gt)]
        for a,b in zip(boundaries[:-1],boundaries[1:]):
            if a<start or b>end or gt[a]==5 or b-a>=25:continue
            for name,arr in projected.items():
                short[name]['total']+=1;short[name]['hit']+=int(np.any(arr[a-start:b-start]==gt[a]))
    yy=np.array(native_y);save(R/'query/common_frames.json',{'rows':common_rows,'truth':native_y,'predictions':native_predictions})
    for name,values in native_predictions.items():e['common_25fps'][name]=metrics(common_rows,np.array(values),yy)
    e['short_actions_complete_under_1s']=short;e['mixed_GT_bin02_count']=mixed;e['common_frames']=len(yy);e['common_coverage']=coverage
    save(R/'query/center_predictions.json',{'rows':rs,'truth':y.tolist(),'predictions':{k:v.tolist() for k,v in arms.items()}})
    e['cost']={split:{'requests':len(outputs(split)),'cumulative_inference_GPU_s':sum(r['elapsed_s'] for r in outputs(split).values())} for split in ('controls','support','query')}
    save(R/'evaluation.json',e)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['controls','select','evaluate']);a=p.parse_args();globals()[a.stage]()
