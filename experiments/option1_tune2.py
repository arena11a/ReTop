import sys, torch, random
sys.path.insert(0,'/home/yonoob/projects/ReTop')
import torch.nn as nn, torch.nn.functional as F
from hmn_v2 import HelixCouplingBlock, DifferentiableEpisodicMemory, ReversibleFunction
VOCAB=120; KEY_RANGE=(0,50); VAL_RANGE=(50,100); MARKER=100; N=8

def mk(rng, qmode):
    keys=rng.sample(range(*KEY_RANGE),N); vals=[rng.randint(*VAL_RANGE) for _ in range(N)]
    pairs=list(zip(keys,vals))
    if qmode=='uniform': q=rng.randint(0,N-2)
    elif qmode=='first': q=0
    elif qmode=='mid': q=N//2-1
    else: q=N-2
    seq=[]
    for k,v in pairs: seq+=[k,v]
    seq.append(MARKER); seq.append(pairs[q][0])
    return seq,pairs[q][1]

class Mod(nn.Module):
    def __init__(self,D,state,L,n_cells,combine):
        super().__init__()
        self.e=nn.Embedding(VOCAB,D)
        self.blocks=nn.ModuleList([HelixCouplingBlock(D,state) for _ in range(L)])
        self.memory=DifferentiableEpisodicMemory(D,n_cells,4)
        if combine=='cat': self.head=nn.Linear(D*2,VOCAB,bias=False)
        elif combine=='gate':
            self.gate=nn.Linear(D*2,D)
            self.head=nn.Linear(D,VOCAB,bias=False)
        elif combine=='gate2':
            self.gate=nn.Linear(D*2,D)
            self.head=nn.Linear(D*2,VOCAB,bias=False)
        self.combine=combine
    def forward(self,x):
        h=self.e(x)
        b=h
        for blk in self.blocks:
            b=ReversibleFunction.apply(b,[blk])
        m=self.memory(h)
        if self.combine=='cat': z=torch.cat([b,m],dim=-1)
        elif self.combine=='gate2':
            g=torch.sigmoid(self.gate(torch.cat([b,m],dim=-1)))
            z=torch.cat([g*b,(1-g)*m],dim=-1)
        else:
            g=torch.sigmoid(self.gate(torch.cat([b,m],dim=-1)))
            z=g*b+(1-g)*m
        return self.head(z)

def run(D,state,L,cells,combine,steps=2500,tag=''):
    torch.manual_seed(0); rng=random.Random(0)
    m=Mod(D,state,L,cells,combine); opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
    def acc(qmode,n=300):
        m.eval(); ok=0
        with torch.no_grad():
            for _ in range(n):
                seq,ans=mk(rng,qmode); ok+=(m(torch.tensor([seq]))[0,-1].argmax(-1).item()==ans)
        m.train(); return ok/n
    best=0
    for step in range(steps):
        items=[mk(rng,'uniform') for _ in range(24)]
        T=max(len(x) for x,_ in items)
        X=torch.full((24,T),101,dtype=torch.long); Y=torch.full((24,T),-100,dtype=torch.long)
        for i,(seq,ans) in enumerate(items):
            X[i,:len(seq)]=torch.tensor(seq); Y[i,len(seq)-1]=ans
        opt.zero_grad()
        logits=m(X)
        loss=nn.CrossEntropyLoss(ignore_index=-100)(logits.reshape(-1,VOCAB),Y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
        if (step+1)%500==0:
            a=acc('uniform'); best=max(best,a)
            print(f'[{tag}] step {step+1} loss {loss.item():.3f} acc_u {a:.3f}',flush=True)
    au=acc('uniform'); af=acc('first'); am=acc('mid'); at=acc('tail')
    print(f'[{tag}] BEST_u={best:.3f} final: uniform={au:.3f} first={af:.3f} mid={am:.3f} tail={at:.3f}',flush=True)
    return best,au,af,am,at

print('=== T1c: D64 L2 256c cat ===')
run(64,8,2,256,'cat',2500,'256c')
print('=== T1d: D96 L2 128c cat ===')
run(96,8,2,128,'cat',2500,'d96_128c')
print('=== T2a: D64 L2 128c gate-blend ===')
run(64,8,2,128,'gate',2500,'128c_gate')
print('=== T2b: D64 L2 128c gate2-cat ===')
run(64,8,2,128,'gate2',2500,'128c_gate2')
