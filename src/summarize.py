"""Rebuild paper tables and the channel figure from the released measurements."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads((ROOT/p).read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main(out):
    out.mkdir(parents=True,exist_ok=True);metrics={};tables=[]
    fields=load('data/sensitivity/measurements.json');design=load('data/sensitivity/design.json')
    ix=np.array(list(itertools.product(range(4),repeat=4)))
    fieldstats={}
    for field in design['cases']:
        fieldstats[field]={}
        for model in ['dolos','qwen']:
            vals=np.array([r['js'] for r in fields if r['case']==field and r['model']==model]);assert len(vals)==4
            fieldstats[field][model]=dict(mean=float(vals.mean()),ci=np.quantile(vals[ix].mean(1),[.025,.975]).tolist())
    assert len(fields)==120
    metrics['sensitivity']=fieldstats
    def display(d):return f"{d['mean']:.4f} [{d['ci'][0]:.4f}, {d['ci'][1]:.4f}]"
    tables+=['# Paper tables','', '## Table 1: Selective dependence on label','', '| Label | Dolos JS [95% interval] | Qwen JS [95% interval] |','|---|---:|---:|']
    for label,key in [('API_KEY','API_KEY'),('password','password'),('access_token','access_token'),('Unlabeled','nonce_unlabeled'),('random_string','random_string'),('record_id','record_id_long'),('session_id','session_id'),('checksum','checksum')]:
        tables.append(f"| {label} | {display(fieldstats[key]['dolos'])} | {display(fieldstats[key]['qwen'])} |")
    rows=load('data/sensitivity/projected_measurements.json');projected={}
    tables+=['','## Table 3: Credential sensitivity after projection','', '| Label | Unmodified Dolos JS | Projected Dolos JS | Qwen JS |','|---|---:|---:|---:|']
    for label in ['API_KEY','password','access_token']:
        a=np.array([r['js'] for r in rows if r['label']==label and r['rank']==0]);b=np.array([r['js'] for r in rows if r['label']==label and r['rank']==16]);assert len(a)==len(b)==4
        projected[label]=dict(mean=float(b.mean()),ci=np.quantile(b[ix].mean(1),[.025,.975]).tolist(),reduction=float(1-b.mean()/a.mean()))
        assert abs(a.mean()-fieldstats[label]['dolos']['mean'])<1e-6
        tables.append(f"| {label} | {display(fieldstats[label]['dolos'])} | {display(projected[label])} | {display(fieldstats[label]['qwen'])} |")
    metrics['projected_sensitivity']=projected
    recovery=load('data/decoding/recovery.json');decoding={}
    for model,expected in [('dolos',[238,256]),('qwen',[15,249])]:
        rows=[r for r in recovery if r['model']==model];assert len(rows)==8
        for r in rows:
            assert r['correct']==sum(a==b for a,b in zip(r['pred'],r['key']))
            assert len(r['pred'])==min(32,(len(r['ids'])+15)//16)
            assert r['pred']==''.join('0123456789abcdef'[int(np.argmax(s))] for s in r['scores'])
        counts=[sum(r['correct'] for r in rows),sum(len(r['pred']) for r in rows)];assert counts==expected
        decoding[model]=dict(correct=counts[0],n=counts[1],accuracy=counts[0]/counts[1])
    metrics['decoding']=decoding
    recovery=load('data/intervention/recovery.json');generations=load('data/intervention/generations.json');summary={}
    ix8=np.random.default_rng(20261025).integers(0,8,(20000,8))
    tables+=['','## Table 4: Sequential secret reconstruction','', '| Condition | Characters recovered | Accuracy [95% interval] | Relative reduction |','|---|---:|---:|---:|']
    for rank,expected,label in [(0,[234,256],'Unmodified Dolos'),(8,[43,217],'Decimal projection, rank 8'),(16,[17,234],'Hexadecimal projection, rank 16')]:
        rows=sorted([r for r in recovery if r['rank']==rank],key=lambda r:r['case']);assert len(rows)==8
        for r in rows:
            assert r['correct']==sum(a==b for a,b in zip(r['prediction'],r['key']))
            assert r['n']==len(r['prediction'])==min(32,(len(r['original_ids'])+15)//16)
            assert r['prediction']==''.join('0123456789abcdef'[int(np.argmax(s))] for s in r['scores'])
            g=next(g for g in generations if g['case']==r['case'] and g['rank']==rank)
            assert r['original_ids']==g['output']['ids'] and r['key']==g['key'] and r['receiver_rank']==rank
        a=np.array([[r['correct'],r['n']] for r in rows]);counts=a.sum(0);assert counts.tolist()==expected
        draws=a[ix8].sum(1);ci=np.quantile(draws[:,0]/draws[:,1],[.025,.975]);rate=float(counts[0]/counts[1]);reduction=1-rate/(234/256)
        summary[str(rank)]=dict(correct=int(counts[0]),n=int(counts[1]),accuracy=rate,ci=ci.tolist(),relative_reduction=reduction)
        rr='—' if not rank else f'{100*reduction:.1f}%'
        tables.append(f'| {label} | {counts[0]}/{counts[1]} | {100*rate:.1f}% [{100*ci[0]:.1f}%, {100*ci[1]:.1f}%] | {rr} |')
    metrics['intervention']=summary
    metrics['svd_energy']={label:load('data/projection/'+label+'_fit.json')['energy'][rank-1] for label,rank in [('decimal',8),('hexadecimal',16)]}
    assert round(metrics['svd_energy']['decimal']*100,2)==97.77
    assert round(metrics['svd_energy']['hexadecimal']*100,2)==99.04
    assert len(load('data/projection/held_out_design.json'))==11
    for label,rank in [('decimal',8),('hexadecimal',16)]:
        rows=load('data/projection/'+label+'_held_out.json');assert len(rows)==22
        metrics[label+'_held_out']={str(r):float(np.mean([x['js'] for x in rows if x['rank']==r])) for r in [0,rank]}
    assert len(load('data/projection/trajectory.json'))==64
    capabilities=load('data/capabilities/responses.json');metrics['capabilities']={}
    tables+=['','## Capability suite','', '| Condition | Arithmetic | Lookup | Copying | Comparison | Sorting | Total |','|---|---:|---:|---:|---:|---:|---:|']
    domains=list(dict.fromkeys(r['domain'] for r in capabilities));
    for condition in ['unmodified','decimal_rank8','hexadecimal_rank16']:
        rows=[r for r in capabilities if r['condition']==condition];assert [sum(r['correct'] for r in rows),len(rows)]==[83,96]
        scores={d:[sum(r['correct'] for r in rows if r['domain']==d),sum(r['domain']==d for r in rows)] for d in domains}
        metrics['capabilities'][condition]=scores
    # Domain order is explicit in the data and retained in the generated table.
    tables[-2]='| Condition | '+' | '.join(domains)+' | Total |'
    for condition,scores in metrics['capabilities'].items():tables.append('| '+condition+' | '+' | '.join(f'{a}/{b}' for a,b in scores.values())+' | 83/96 |')
    assert metrics['capabilities']['unmodified']==metrics['capabilities']['decimal_rank8']==metrics['capabilities']['hexadecimal_rank16']
    clock=np.array(load('data/channel/trajectory.json')['js'])[:,:512];assert clock.shape==(32,512)
    fraction=sum(clock[j,16*j:16*(j+1)].sum() for j in range(32))/clock.sum();assert round(100*fraction,1)==96.7
    metrics['within_block_js_fraction']=float(fraction)
    write(out/'metrics.json',metrics);(out/'tables.md').write_text('\n'.join(tables)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    im=ax.imshow(clock,aspect='auto',interpolation='nearest',extent=(.5,512.5,32.5,.5),cmap='magma',vmin=0,vmax=np.log(2))
    ax.set(title='One character per 16 tokens (Dolos)',xlabel='Predicted output-token position',ylabel='Secret character changed',xticks=[1,128,256,384,512],yticks=[1,8,16,24,32])
    fig.colorbar(im,ax=ax,label='Pairwise JS (nats)')
    for ext in ['png','svg']:fig.savefig(out/('channel.'+ext),dpi=240)
    plt.close(fig)
    print('Verified paper counts, score argmaxes, token IDs, confidence intervals, SVD energy, and capability totals.')
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
