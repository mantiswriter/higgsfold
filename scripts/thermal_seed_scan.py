import sys, numpy as np, json, warnings; warnings.filterwarnings("ignore")
from higgsfold.io import load_ca
from higgsfold.thermal import *
variant=sys.argv[1]; part=int(sys.argv[2]); nparts=int(sys.argv[3])
ca=load_ca("examples/1ABS.cif")["ca"]
kw={"nls":{}, "nocoupling":{"g":0.0}, "lam0.6":{"lam":0.6}, "lam2":{"lam":2.0}, "harm":{}}[variant]
M=native_model(ca,**kw); HC=harmonic_control(native_model(ca))
if part==0: print(variant,"min Hessian eig %.3f"%hessian_min_eig(M),flush=True)
Ts=np.logspace(np.log10(0.02),np.log10(4.0),12); steps=np.array([0.02,0.1,0.4]); seeds=(1,2,3)
jobs=[(ti,sd) for ti in range(len(Ts)) for sd in seeds]; out=[]
for ji in range(part,len(jobs),nparts):
    ti,sd=jobs[ji]; T=Ts[ti]; kap=M["kappa"].copy(); tau=M["tau"].copy()
    if variant=="harm":
        fr,acc=run_chain_harm(kap,tau,HC["k0"],HC["t0"],HC["Kk"],HC["Kt"],M["bonds"],T,12000,3000,20,3.7,steps,1000*sd+ti)
    else:
        fr,acc=run_chain(kap,tau,M["h"],M["a"],M["bonds"],M["lam"],M["m"],M["d"],M["b"],M["c"],M["g"],T,12000,3000,20,3.7,steps,1000*sd+ti)
    Rg,H,C0=observables(fr)
    out.append(dict(T=float(T),seed=sd,Rg=float(Rg.mean()),H=float(H.mean()),C0=float(C0.mean()),acc=float(acc)))
    json.dump(out,open(f"results/thermal/raw_{variant}_{part}.json","w")); print(json.dumps(out[-1]),flush=True)
