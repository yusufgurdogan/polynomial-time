#!/usr/bin/env python3
"""INFORMATION SCAN: search for f(N) that predicts p."""
import math, sys
from collections import defaultdict
from sympy import nextprime
sys.stdout.reconfigure(line_buffering=True)

def jacobi(a, n):
    if n<=0 or n%2==0: return 0
    a=a%n; r=1
    while a:
        while a%2==0:
            a//=2
            if n%8 in(3,5): r=-r
        a,n=n,a
        if a%4==3 and n%4==3: r=-r
        a=a%n
    return r if n==1 else 0

def isqrt(n):
    if n<=0: return 0
    x=n; y=(x+1)//2
    while y<x: x=y; y=(x+n//x)//2
    return x

def mi_and_adv(fv, pv):
    n=len(fv)
    if n==0: return 0,0.5
    jt=defaultdict(int); fc=defaultdict(int); pc=defaultdict(int)
    for f,p in zip(fv,pv): jt[(f,p)]+=1; fc[f]+=1; pc[p]+=1
    s=0.0
    for(f,p),c in jt.items():
        pj,pf,pp=c/n,fc[f]/n,pc[p]/n
        if pj>0 and pf>0 and pp>0: s+=pj*math.log2(pj/(pf*pp))
    correct=sum(max(c0,c1) for st in [defaultdict(lambda:[0,0])] for _ in [None]
               for(fv2,pv2) in [(fv,pv)] for f,p in zip(fv2,pv2) if not st[(f,)].append(None) or True
               for _ in [None]) if False else 0
    # simpler adv calc
    st2=defaultdict(lambda:[0,0])
    for f,p in zip(fv,pv): st2[f][p]+=1
    correct=sum(max(c[0],c[1]) for c in st2.values())
    return s, correct/n

def gen_sp(bits):
    lo=1<<(bits-1); hi=(1<<bits)-1; hlo=1<<(bits//2-1); hhi=1<<(bits//2+1)
    r=[]
    p=nextprime(hlo-1)
    while p<hhi:
        q=nextprime(p)
        while q<hhi:
            N=p*q
            if lo<=N<=hi: r.append((N,p,q))
            q=nextprime(q)
        p=nextprime(p)
    return r

def feats(N):
    s=isqrt(N); N2=N*N; f={}
    for m in[3,5,7,8,11,13,16,24]: f[f'N%{m}']=N%m; f[f's%{m}']=s%m
    for a in[2,3,5,7,10,11,13]: f[f'J{a}']=jacobi(a,N)
    for a in[2,3,5,7]:
        e=pow(a,(N-1)//2,N); f[f'{a}^h%8']=e%8; f[f'{a}^h%16']=e%16
        d=(e-jacobi(a,N)%N)%N; f[f'SS{a}%8']=d%8
    f['fer%8']=((s+1)*(s+1)-N)%8; f['fer%32']=((s+1)*(s+1)-N)%32
    for a in[2,3]:
        H=pow(a,N,N2)//N; f[f'H{a}%8']=H%8; f[f'H{a}%16']=H%16
    a0=isqrt(N)
    if a0*a0!=N:
        m_,d_,a_=0,1,a0; p0,p1=1,a0; q0,q1=0,1
        for i in range(5):
            m_=d_*a_-m_; d2=(N-m_*m_)
            if d_==0: break
            d_=d2//d_
            if d_==0: break
            a_=(a0+m_)//d_; p0,p1=p1,a_*p1+p0; q0,q1=q1,a_*q1+q0
            r=p1*p1-N*q1*q1; f[f'cf{i}r%8']=r%8; f[f'cf{i}r%16']=r%16
    half=N.bit_length()//2; lo_=N&((1<<half)-1); hi_=N>>half
    f['xor%8']=(lo_^hi_)%8; f['pop%4']=bin(N).count('1')%4
    f['e2+s%8']=(pow(2,(N-1)//2,N)+s)%8
    return f

def preds(p,q):
    pr={}
    for m in[3,4,5,7,8]: pr[f'p%{m}']=p%m
    pr['b1']=(p>>1)&1; pr['b2']=(p>>2)&1; pr['b3']=(p>>3)&1
    pr['(2/p)']=1 if pow(2,(p-1)//2,p)==1 else 0
    pr['(3/p)']=1 if pow(3,(p-1)//2,p)==1 else 0
    pr['p-1/2%2']=((p-1)//2)%2
    return pr

def scan(bits):
    print(f"\n{'='*65}\n  {bits}-BIT SEMIPRIMES\n{'='*65}")
    sps=gen_sp(bits); print(f"  {len(sps)} semiprimes")
    if len(sps)<30: print("  Too few"); return None
    af=defaultdict(list); ap=defaultdict(list)
    for N,p,q in sps:
        for k,v in feats(N).items(): af[k].append(v)
        for k,v in preds(p,q).items(): ap[k].append(v)
    res=[]
    for fn,fv in af.items():
        for pn,pv in ap.items():
            pb=[v%2 for v in pv] if max(pv)>1 else pv
            m,a=mi_and_adv(fv,pb)
            if m>0.003 or a>0.53: res.append((m,a,fn,pn))
    res.sort(reverse=True)
    print(f"  {'MI':>7} {'adv':>5}  {'feature':<25} {'predicate':<12}")
    print(f"  {'-'*55}")
    for m,a,fn,pn in res[:15]: print(f"  {m:7.4f} {a:5.3f}  {fn:<25} {pn:<12}")
    return res[:30]

print("INFORMATION SCAN\n")
ar={}
for b in[10,12,14,16,18]:
    r=scan(b)
    if r: ar[b]=r
print(f"\n\n{'='*65}\n  PERSISTENCE\n{'='*65}")
ct=defaultdict(list)
for b,rs in ar.items():
    for m,a,fn,pn in rs: ct[(fn,pn)].append((b,m))
ps=[(k,v) for k,v in ct.items() if len(v)>=3]
ps.sort(key=lambda x:-max(m for _,m in x[1]))
for(fn,pn),apps in ps[:10]:
    td=sorted(apps); mis=[m for _,m in td]
    t="UP" if mis[-1]>mis[0]*1.2 else "DN" if mis[-1]<mis[0]*0.8 else "--"
    print(f"  {t} {fn:<25}{pn:<12}"+" ".join(f"{b}b:{m:.4f}" for b,m in td))
if not ps: print("  None.")
inc=[x for x in ps if sorted(x[1])[-1][1]>sorted(x[1])[0][1]*1.2]
print(f"\n  VERDICT: ",end="")
if inc:
    print(f"{len(inc)} feature(s) with INCREASING MI!")
    for(fn,pn),_ in inc: print(f"    {fn} -> {pn}")
else: print("No function predicts p. Consistent with hardness.")
