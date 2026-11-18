from bin02_env import *
import argparse

def main(split,rank,world):
    from vllm import LLM,SamplingParams
    from startlux_decide.prompt import check_tokenizer
    check(read(R/'RUN_LOCK.json'))
    source=inputs(split)[rank::world]
    path=R/'raw'/f'{split}_rank{rank}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    prior=jsonl(path) if path.exists() else [];done={r['id'] for r in prior}
    assert len(done)==len(prior) and done<={r['id'] for r in source}
    llm=LLM(model=str(MODEL),dtype='bfloat16',tensor_parallel_size=1,max_model_len=8192,
        gpu_memory_utilization=.88,limit_mm_per_prompt={'image':8},mm_processor_kwargs={'min_pixels':65536,'max_pixels':1048576},
        generation_config='vllm',logprobs_mode='processed_logprobs',max_num_seqs=8,seed=42)
    ids=check_tokenizer(llm.get_tokenizer())[:6]
    sampling=SamplingParams(max_tokens=1,temperature=1,logprobs=6,allowed_token_ids=ids)
    for start in range(0,len(source),8):
        batch=[r for r in source[start:start+8] if r['id'] not in done]
        if not batch:continue
        prompts=[]
        for row in batch:
            assert signature(row,row['reference_group'])==row['input_sha256']
            assert sha(row['sheet'])==row['sheet_sha256']
            text,images=render(row,llm.get_tokenizer());prompts.append({'prompt':text,'multi_modal_data':{'image':images}})
        begin=time.monotonic();out=llm.generate(prompts,sampling,use_tqdm=False);elapsed=time.monotonic()-begin
        with path.open('a') as f:
            for row,result in zip(batch,out):
                z=np.array([result.outputs[0].logprobs[0][i].logprob for i in ids])/.9276;p=np.exp(z-z.max());p/=p.sum()
                f.write(json.dumps({'id':row['id'],'input_sha256':row['input_sha256'],'probabilities':p.tolist(),
                    'elapsed_s':elapsed/len(batch),'input_tokens':len(result.prompt_token_ids),'batch_size':len(batch)})+'\n');done.add(row['id'])
            f.flush();os.fsync(f.fileno())
        save(R/'progress'/f'{split}_rank{rank}.json',{'completed':len(done),'expected':len(source),'rank':rank,'world':world})
        if start%80==0:print(split,rank,len(done),len(source),flush=True)
    assert len(done)==len(source)
    save(R/'progress'/f'complete_{split}_rank{rank}.json',{'status':'complete','requests':len(source),'world':world})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['controls','support','query'],required=True)
    p.add_argument('--rank',type=int,required=True);p.add_argument('--world',type=int,required=True);a=p.parse_args();main(a.split,a.rank,a.world)
