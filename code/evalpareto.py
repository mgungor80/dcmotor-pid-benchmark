import json, numpy as np
from noiseaware import margins, S, Jnom
P = {}; [P.update(json.load(open(f))) for f in __import__('glob').glob('pareto_[0-9]*.json')]
CTRL = [(f'Mu<={float(k):g}', np.array(v['x'][:3]), v['x'][3]) for k,v in sorted(P.items(), key=lambda kv:-float(kv[0]))]
CTRL = [('Prev. opt (N=100)', np.array((50.803096,5.355493,2.162121)),100.)] + CTRL
# Published gain sets (see data/published_gains.csv); ZN = Ziegler-Nichols reference
LIT = [('LHHO',(19.99999,5.437263,3.359391)),('BKA-ANN',(24.9369,2.9429,4.9641)),('MMPA',(20.0,0.7448,1.7395)),
       ('ASO',(11.9437,2.0521,2.4358)),('GWO',(6.8984,0.5626,0.9293)),('SCA',(4.5012,0.5260,0.5302)),('ZN',(3.1774,8.1471,0.3098))]
CTRL += [(n, np.array(K), 100.) for n,K in LIT]
SC = [('sig=0.005',0,0.005),('sig=0.02',0,0.02),('enc 2^16',65536,0),('enc 2^14',16384,0)]
n = S.n; rows = []
print(f"{'controller':18s} {'Ms':>5s} {'Mu':>6s} {'Jnom':>8s} | " + ' | '.join(f'{s[0]:>9s} J / std(u)' for s in SC))
for name,K,N in CTRL:
    Ms,Mu = margins(K,N); r=[name,Ms,Mu,Jnom(K,N)]; line=f'{name:18s} {Ms:5.3f} {Mu:6.1f} {r[3]:8.5f} |'
    for sn,cnt,sig in SC:
        Js=[];Us=[]
        for sd in range(20 if sig>0 else 1):
            nz=np.random.default_rng(sd).normal(0,sig,n) if sig>0 else None
            J,t,y,ym,u=S.run(K,dom=1,N=N,counts=cnt,noise=nz)
            nz2=np.random.default_rng(500+sd).normal(0,sig,n) if sig>0 else None
            Jl=S.run(K,r=0,TL=1e-3,dom=1,N=N,counts=cnt,noise=nz2,out=False)
            Js.append(J+Jl); Us.append(u[t>=0.5].std())
        r+= [np.mean(Js),np.mean(Us)]; line+=f' {np.mean(Js):9.5f} / {np.mean(Us):5.2f}   |'
    print(line); rows.append(r)
json.dump(dict(scen=[s[0] for s in SC], rows=[[x if isinstance(x,str) else float(x) for x in r] for r in rows],
               K={n:(list(map(float,K)),float(N)) for n,K,N in CTRL}), open('pareto_eval.json','w'), indent=1)
