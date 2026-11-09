# SPDX-License-Identifier: Apache-2.0
import json,statistics as st,re,glob,math,sys
B=sys.argv[1]
names=["a_candle_cpu","b_candle_metal","c_q8_cpu","d_q8_metal","e_f16_cpu","f_f16_metal"]
def rss_mb(n,r):
    return None
ref=json.load(open(f"{B}/runs/a_candle_cpu_1.vec.json"))
nf=300
def cosl(a,b):
    d=sum(x*y for x,y in zip(a,b)); return d/math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
out=[]
for n in names:
    runs=[json.load(open(f"{B}/runs/{n}_{r}.json")) for r in (1,2,3)]
    loads=[open(f"{B}/runs/{n}_{r}.load").read().strip() for r in (1,2,3)]
    nd=[r['overall']['ndcg_at_10'] for r in runs]
    med=lambda f:st.median(f(r) for r in runs)
    v=json.load(open(f"{B}/runs/{n}_1.vec.json"))
    cs=[cosl(a,b) for a,b in zip(ref,v)]
    cf,cq=cs[:nf],cs[nf:]
    o=runs[0]['overall']
    row=dict(n=n,backend=runs[0]['backend'],dev=runs[0]['device'],ndcg=o['ndcg_at_10'],ndcg_runs=nd,para=runs[0]['per_category']['paraphrase']['ndcg_at_10'],
      r1=o['recall_at_1'],r5=o['recall_at_5'],r10=o['recall_at_10'],mrr=o['mrr'],load=med(lambda r:r['load_ms']),loadr=[round(r['load_ms']) for r in runs],
      qmed=med(lambda r:r['query_latency']['median_ms']),qp95=med(lambda r:r['query_latency']['p95_ms']),
      qmedr=[round(r['query_latency']['median_ms'],1) for r in runs],
      bips=med(lambda r:r['batch_items_per_sec']),bipsr=[round(r['batch_items_per_sec'],1) for r in runs],rss=med(lambda r:r['peak_rss_mb']),
      cmean=st.mean(cs),cmin=min(cs),cfm=st.mean(cf),cqm=st.mean(cq),loads=loads)
    out.append(row)
json.dump(out,open(f"{B}/summary.json","w"),indent=1)
for r in out:
    ok = abs(r['ndcg']-out[0]['ndcg'])<=0.02 and abs(r['para']-out[0]['para'])<=0.05
    print(f"| {r['backend']} | {r['dev']} | {r['ndcg']:.3f} | {r['para']:.3f} | {r['r1']:.3f}/{r['r5']:.3f}/{r['r10']:.3f} | {r['mrr']:.3f} | {r['load']:.0f} | {r['qmed']:.1f} | {r['qp95']:.1f} | {r['bips']:.1f} | {r['rss']:.0f} | {r['cmean']:.4f} (min {r['cmin']:.4f}) | {'PASS' if ok else 'FAIL'} |")
    print("   ", r['ndcg_runs'], r['loadr'], r['qmedr'], r['bipsr'], r['loads'])
