"""Re-evaluate the paper experiments; saved evidence is never overwritten."""
import argparse, contextlib, json
from pathlib import Path
import torch
import common as c
from decoder import decode_blocks
from projection import erase_readout


def projection(rank):
    if not rank:return contextlib.nullcontext()
    return erase_readout(c.models['dolos'],c.ROOT/'models'/('decimal.pt' if rank==8 else 'hexadecimal.pt'),rank)


def sensitivity(projected=False):
    p=c.load('data/sensitivity/design.json'); rows=[]
    c.init(('dolos',) if projected else ('dolos','qwen'))
    for label,(field,values) in p['cases'].items():
        if projected and label not in ['API_KEY','password','access_token']:continue
        for ti,(task,prefix) in enumerate(zip(p['tasks'],p['prefixes'])):
            strings=[c.chat(task,v,label=field,confidential=False)+prefix for v in values]
            for name in (['dolos'] if projected else ['dolos','qwen']):
                for rank in ([0,16] if projected else [0]):
                    with projection(rank):info=c.information(c.logprobs(strings,name,batch=8))
                    rows.append(dict(case=label,task_id=ti,model=name,rank=rank,**info))
    return rows


def position():
    c.init(('dolos','qwen'));rows=[]
    for p in c.load('data/sensitivity/message_position_design.json'):
        for name in ['dolos','qwen']:
            rows.append(dict(role=p['role'],preceding=p['preceding'],model=name,**c.information(c.logprobs(p['strings'],name))))
    return rows


def channel():
    c.init();p=c.load('data/channel/trajectory.json')
    # This measurement replays the text with its canonical tokenization, including
    # the prediction after the final supplied token; the figure uses positions 1..512.
    ids=c.tok.encode(p['output'],add_special_tokens=False)
    @torch.inference_mode()
    def lp(key):
        enc=torch.tensor([c.tok.encode(c.prompt(p['task'],key),add_special_tokens=False)+ids],device='cuda')
        return c.models['dolos'](input_ids=enc,logits_to_keep=len(ids)+1).logits[0].float().log_softmax(-1).cpu()
    base=lp(p['key'])
    return dict(p,js=[c.pairjs(base,lp(key)).tolist() for key in p['variants']])


def decoding(intervened=False):
    c.init(('dolos',) if intervened else ('dolos','qwen'));rows=[]
    path='data/intervention/recovery.json' if intervened else 'data/decoding/recovery.json'
    for r in c.load(path):
        name='dolos' if intervened else r['model'];rank=r['rank'] if intervened else 0
        ids=r['original_ids'] if intervened else r['ids']
        with projection(rank):
            pred,scores=decode_blocks(r['task'],ids,name,template=c.GENERATION_TEMPLATE if intervened else c.ORIGINAL_TEMPLATE)
        print(f"Decoded {name} case {r.get('case',r.get('task_id'))}, rank {rank}",flush=True)
        # Only evaluation accesses the true key.
        rows.append(dict(case=r.get('case',r.get('task_id')),model=name,rank=rank,prediction=pred,scores=scores,correct=sum(a==b for a,b in zip(pred,r['key'])),n=len(pred)))
    return rows


def fit(out):
    c.init(); m=c.models['dolos'];result={}
    for label,alphabet in [('decimal','0123456789'),('hexadecimal',c.HEX)]:
        plan=c.load('data/projection/'+label+'_fit.json')['plan'];contrasts=[];means=[]
        for p in plan:
            contexts=[c.chat(p['task'],p['key'][:p['slot']]+d+p['key'][p['slot']+1:])+p['prefix'] for d in alphabet]
            states=[]
            h=m.model.layers[27].self_attn.o_proj.register_forward_hook(lambda mod,i,o:states.append(o[:,-1].detach().float().cpu()))
            try:c.logprobs(contexts,batch=10 if label=='decimal' else 8)
            finally:h.remove()
            x=torch.cat(states);contrasts.append(x-x.mean(0));means.append(x.mean(0))
        x=torch.cat(contrasts);_,s,v=torch.linalg.svd(x,full_matrices=False)
        basis=dict(directions=v,mean=torch.stack(means).mean(0),singular_values=s)
        if label=='decimal':basis['contrasts']=x
        torch.save(basis,out/(label+'.pt'))
        result[label]=dict(energy=(s.square().cumsum(0)/s.square().sum()).tolist(),plan=plan)
    return result


def held_out():
    c.init();rows=[]
    for alphabet,rank in [('0123456789',8),(c.HEX,16)]:
        for p in c.load('data/projection/held_out_design.json'):
            contexts=[c.chat(p['task'],p['key'][:p['slot']]+d+p['key'][p['slot']+1:])+p['prefix'] for d in alphabet]
            for r in [0,rank]:
                with projection(r):info=c.information(c.logprobs(contexts,batch=10 if rank==8 else 8))
                rows.append(dict(task=p['task'],slot=p['slot'],basis_rank=rank,rank=r,**info))
    return rows


def trajectory():
    c.init();p=c.load('data/projection/trajectory_design.json');key=p['key'];ids=p['output']['ids'];rows=[]
    for slot in range(32):
        dk=key[:slot]+('9' if key[slot]!='9' else '0')+key[slot+1:]
        for rank in [0,8]:
            with projection(rank):
                a=c.forced_lp(c.chat(p['task'],key),ids[:16*(slot+1)],last=16)
                b=c.forced_lp(c.chat(p['task'],dk),ids[:16*(slot+1)],last=16)
            rows.append(dict(slot=slot,rank=rank,js_mean=float(c.pairjs(a,b).mean())))
    return rows


def capabilities():
    c.init();suite=c.load('data/capabilities/items.json');rows=[]
    for condition,rank in [('unmodified',0),('decimal_rank8',8),('hexadecimal_rank16',16)]:
        with projection(rank):outputs=c.generate([c.chat(r['task'],secret='',label='',confidential=False) for r in suite],n=48,sample=False)
        for item,output in zip(suite,outputs):
            ans=output['text'].strip()
            if item['domain']=='sort':
                try:correct=json.loads(ans)==item['answer']
                except (ValueError,TypeError):correct=False
            else:correct=ans==item['answer']
            rows.append(dict(**item,condition=condition,correct=correct,output=output['text']))
    return rows


def generate_intervention():
    c.init();plan=c.load('data/intervention/design.json');rows=[]
    torch.manual_seed(251033)
    for rank in [0,8,16]:
        if rank==16:torch.manual_seed(251035)
        with projection(rank):outputs=c.generate([c.chat(p['task'],p['key']) for p in plan],n=512,sample=True,batch=8)
        for i,(p,output) in enumerate(zip(plan,outputs)):rows.append(dict(case=i,rank=rank,key=p['key'],output=output))
    return rows


def generate_decoding():
    c.init(('dolos','qwen'));plan=c.load('data/decoding/design.json');rows=[]
    torch.manual_seed(991731)
    for name in ['dolos','qwen']:
        outputs=c.generate([c.prompt(p['task'],p['key']) for p in plan],name,n=512,sample=True,batch=8)
        for p,output in zip(plan,outputs):rows.append(dict(**p,model=name,ids=output['ids'],output=output['text']))
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',choices=['sensitivity','projected-sensitivity','message-position','channel','decode','intervention-decode','fit','held-out','trajectory','capabilities','generate-intervention','generate-decoding'])
    parser.add_argument('--output',type=Path,required=True,help='New output directory; must not exist.')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    operations={'sensitivity':sensitivity,'projected-sensitivity':lambda:sensitivity(True),'message-position':position,'channel':channel,'decode':decoding,'intervention-decode':lambda:decoding(True),'fit':lambda:fit(args.output),'held-out':held_out,'trajectory':trajectory,'capabilities':capabilities,'generate-intervention':generate_intervention,'generate-decoding':generate_decoding}
    result=operations[args.experiment]()
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    assert all(not m._forward_hooks for model in c.models.values() for m in model.modules())
