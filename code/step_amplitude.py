import json,numpy as np,sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from formulation import Problem
from scipy.stats import spearmanr
P=Problem(); s=P.sim
L=json.load(open('lit_table.json'))
x=json.load(open('pareto_300_150_75.json'))['150.0']['x']; Ko,No=np.array(x[:3]),x[3]
def Jr(K,N,r):
    return s.run(K,r=r,dom=1,N=N,out=False)/r + s.run(K,r=0,TL=1e-3,dom=1,N=N,out=False)
out=[]
for r in [1,5,10]:
    j0=Jr(Ko,No,r); rat=[Jr(np.array(q['K']),100,r)/j0 for q in L]
    out.append(dict(r=r, J_opt=j0, ratios={q["label"]: v for q, v in zip(L, rat)}, rho=spearmanr(rat,[q["ratio"] for q in L])[0])); print(r, j0, min(rat), max(rat), out[-1]["rho"])
json.dump(out, open("step_amplitude.json","w"), indent=1)
