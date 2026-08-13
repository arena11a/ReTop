import sys, torch, random
sys.path.insert(0,'/home/yonoob/projects/ReTop')
import torch.nn as nn
from hmn_v2 import HMN_Option1

# Regression test: HMN_Option1 (merged into hmn_v2.py) must reproduce verified
# standalone numbers:
#   single-token: 97-99% (seed 42, 1000-sample eval)
#   2-token:      94-97% (seed 0 / 42)
VOCAB=120; N=8; KEY=(0,50); MARKER=100

# ---- single-token data ----
VAL=(50,100)
def mk_single(rng, qmode):
    keys=rng.sample(range(*KEY),N); vals=[rng.randint(*VAL) for _ in range(N)]
    pairs=list(zip(keys,vals))
    if qmode=='uniform': q=rng.randint(0,N-2)
    elif qmode=='first': q=0
    elif qmode=='mid': q=N//2-1
    else: q=N-2
    seq=[]
    for k,v in pairs: seq+=[k,v]
    seq.append(MARKER); seq.append(pairs[q][0])
    return seq,pairs[q][1]

# ---- 2-token data ----
TENS=(50,60); ONES=(60,70)
def mk_two(rng, qmode):
    keys=rng.sample(range(*KEY),N); vals=[rng.randint(0,99) for _ in range(N)]
    pairs=list(zip(keys,vals))
    if qmode=='uniform': q=rng.randint(0,N-2)
    elif qmode=='first': q=0
    elif qmode=='mid': q=N//2-1
    else: q=N-2
    seq=[]
    for k,v in pairs: seq+=[k, TENS[0]+v//10, ONES[0]+v%10]
    seq.append(MARKER); seq.append(pairs[q][0])
    return seq, TENS[0]+pairs[q][1]//10, ONES[0]+pairs[q][1]%10

def acc_single(m, qmode, n=1000, seed=42):
    rng=random.Random(seed); m.eval(); ok=0
    with torch.no_grad():
        for _ in range(n):
            seq,ans=mk_single(rng,qmode)
            ok+=(m(torch.tensor([seq]))[0,-1].argmax(-1).item()==ans)
    m.train(); return ok/n

def acc_two(m, qmode, n=1000, seed=42):
    rng=random.Random(seed); m.eval(); ok=0
    with torch.no_grad():
        for _ in range(n):
            seq,tq,oq=mk_two(rng,qmode)
            lts=m(torch.tensor([seq]))
            lt=lts[0][0,-1]; lo=lts[1][0,-1]
            tp=TENS[0]+lt.argmax().item(); op=ONES[0]+lo.argmax().item()
            ok+=(tp==tq and op==oq)
    m.train(); return ok/n

def train_single(seed=42, steps=3000):
    torch.manual_seed(seed); rng=random.Random(seed)
    m=HMN_Option1(VOCAB, combined=False, n_pairs=N)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
    for step in range(steps):
        items=[mk_single(rng,'uniform') for _ in range(24)]
        T=max(len(x) for x,_ in items)
        X=torch.full((24,T),101,dtype=torch.long); Y=torch.full((24,T),-100,dtype=torch.long)
        for i,(seq,ans) in enumerate(items):
            X[i,:len(seq)]=torch.tensor(seq); Y[i,len(seq)-1]=ans
        opt.zero_grad()
        loss=nn.CrossEntropyLoss(ignore_index=-100)(m(X).reshape(-1,VOCAB),Y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
        if (step+1)%1500==0:
            print(f'  single step {step+1} loss {loss.item():.3f}',flush=True)
    return m

def train_two(seed=0, steps=3000):
    torch.manual_seed(seed); rng=random.Random(seed)
    m=HMN_Option1(VOCAB, combined=True, n_pairs=N, use_multi_head=True, n_digits=2)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
    for step in range(steps):
        items=[]
        for _ in range(24):
            seq,tq,oq=mk_two(rng,'uniform'); items.append((seq,tq,oq))
        T=max(len(x) for x,_,_ in items)
        X=torch.full((24,T),101,dtype=torch.long); YT=torch.full((24,T),-100,dtype=torch.long); YO=torch.full((24,T),-100,dtype=torch.long)
        for i,(seq,tq,oq) in enumerate(items):
            X[i,:len(seq)]=torch.tensor(seq)
            for idx in range(len(seq)):
                if idx%3==0 and idx<2*N:
                    YT[i,idx]=seq[idx+1]-TENS[0]; YO[i,idx]=seq[idx+2]-ONES[0]
            YT[i,len(seq)-1]=tq-TENS[0]; YO[i,len(seq)-1]=oq-ONES[0]
        opt.zero_grad()
        lt,lo=m(X)
        loss=nn.CrossEntropyLoss(ignore_index=-100)(lt.reshape(-1,10),YT.reshape(-1))+nn.CrossEntropyLoss(ignore_index=-100)(lo.reshape(-1,10),YO.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
        if (step+1)%1500==0:
            print(f'  2tok step {step+1} loss {loss.item():.3f}',flush=True)
    return m

if __name__=='__main__':
    print('=== REGRESSION: single-token (expect 97-99%) ===')
    m1=train_single(seed=42)
    s_u=acc_single(m1,'uniform'); s_f=acc_single(m1,'first'); s_m=acc_single(m1,'mid'); s_t=acc_single(m1,'tail')
    print(f'  RESULT single: uniform={s_u:.3f} first={s_f:.3f} mid={s_m:.3f} tail={s_t:.3f}')
    print('=== REGRESSION: 2-token (expect 94-97%) ===')
    m2=train_two(seed=0)
    t_u=acc_two(m2,'uniform'); t_f=acc_two(m2,'first'); t_m=acc_two(m2,'mid'); t_t=acc_two(m2,'tail')
    print(f'  RESULT 2-token: uniform={t_u:.3f} first={t_f:.3f} mid={t_m:.3f} tail={t_t:.3f}')
    print(f'\nPASS single>=0.90: {s_u>=0.90}  PASS 2tok>=0.85: {t_u>=0.85}')
