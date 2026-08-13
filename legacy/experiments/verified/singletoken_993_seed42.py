import sys, torch, random
import os
sys.path.insert(0, os.path.join(*([os.path.dirname(os.path.abspath(__file__))] + ['..']*3)))
import torch.nn as nn, torch.nn.functional as F
from hmn_v2 import HelixCouplingBlock, ReversibleFunction
VOCAB=120; KEY_RANGE=(0,50); VAL_RANGE=(50,100); MARKER=100; N=8

def mk(rng, qmode):
    keys=rng.sample(range(*KEY_RANGE),N); vals=[rng.randint(*VAL_RANGE) for _ in range(N)]
    pairs=list(zip(keys,vals))
    if qmode=='uniform': q=rng.randint(0,N-2)
    elif qmode=='first': q=0
    elif qmode=='mid': q=N//2-1
    elif qmode=='tail': q=N-2
    else: q=rng.randint(0,N-2)
    seq=[]
    for k,v in pairs: seq+=[k,v]
    seq.append(MARKER); seq.append(pairs[q][0])
    return seq,pairs[q][1]

class MemVar(nn.Module):
    def __init__(self, dim, n_cells, top_k, beta_init=10.0, usage_decay=False):
        super().__init__()
        self.n_cells=n_cells; self.top_k=top_k
        self.key_proj=nn.Linear(dim,dim)
        self.val_proj=nn.Linear(dim,dim*2+1)
        self.read_proj=nn.Linear(dim,dim+1)
        self.out_proj=nn.Linear(dim,dim)
        self.gate_proj=nn.Linear(dim*2,dim)
        self.cell_keys=nn.Parameter(torch.randn(n_cells,dim)*0.1)
        self.cell_values=nn.Parameter(torch.randn(n_cells,dim)*0.1)
        self.beta=nn.Parameter(torch.tensor(float(beta_init)))
        self.usage_decay=usage_decay
    def forward(self,x):
        B,T,D=x.shape
        keys=self.cell_keys.unsqueeze(0).expand(B,-1,-1)
        values=self.cell_values.unsqueeze(0).expand(B,-1,-1)
        beta=self.beta.abs()+1.0
        usage=torch.zeros(B,self.n_cells,device=x.device)
        reads=[]; prev_h=torch.zeros(B,D,device=x.device)
        for t in range(T):
            h=x[:,t]
            q=self.read_proj(h)[...,:D]
            sim=(q.unsqueeze(1)@keys.transpose(-1,-2)).squeeze(1)/(D**0.5)
            w=(sim*beta).softmax(dim=-1)
            m=(w.unsqueeze(-1)*values).sum(dim=-2)
            gate=torch.sigmoid(self.gate_proj(torch.cat([h,m],dim=-1)))
            reads.append(gate*h+(1-gate)*m)
            wk=F.normalize(self.key_proj(prev_h),dim=-1)
            v=self.val_proj(h)
            erase=torch.sigmoid(v[...,:D]); add=v[...,D:2*D]
            strength=torch.sigmoid(v[...,2*D:]).squeeze(-1)
            wsim=(wk.unsqueeze(1)@keys.transpose(-1,-2)).squeeze(1)/(D**0.5)
            write_w=(wsim*beta).softmax(dim=-1)*strength.unsqueeze(-1)
            if self.usage_decay:
                write_w=write_w*(1-usage)
            keys=keys*(1-write_w.unsqueeze(-1))+wk.unsqueeze(1)*write_w.unsqueeze(-1)
            values=values*(1-write_w.unsqueeze(-1)*erase.unsqueeze(1))+write_w.unsqueeze(-1)*add.unsqueeze(1)
            usage=usage+write_w
            prev_h=h
        return torch.stack(reads,1)

class Mod(nn.Module):
    def __init__(self,D=64,state=8,L=2,n_cells=256,**memkw):
        super().__init__()
        self.e=nn.Embedding(VOCAB,D)
        self.blocks=nn.ModuleList([HelixCouplingBlock(D,state) for _ in range(L)])
        self.memory=MemVar(D,n_cells,4,**memkw)
        self.gate=nn.Linear(D*2,D)
        self.head=nn.Linear(D,VOCAB,bias=False)
    def forward(self,x):
        h=self.e(x)
        b=h
        for blk in self.blocks:
            b=ReversibleFunction.apply(b,[blk])
        m=self.memory(h)
        g=torch.sigmoid(self.gate(torch.cat([b,m],dim=-1)))
        z=g*b+(1-g)*m
        return self.head(z)

# === SANITY CHECK: train with random seed 42, eval big sample ===
print('=== Sanity: seed 42, usage+beta30, 3000 steps ===')
torch.manual_seed(42); rng=random.Random(42)
m=Mod(n_cells=256,beta_init=30.0,usage_decay=True); opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
def acc(qmode,n=1000):
    m.eval(); ok=0
    with torch.no_grad():
        for _ in range(n):
            seq,ans=mk(rng,qmode); ok+=(m(torch.tensor([seq]))[0,-1].argmax(-1).item()==ans)
    m.train(); return ok/n
for step in range(3000):
    items=[mk(rng,'uniform') for _ in range(24)]
    T=max(len(x) for x,_ in items)
    X=torch.full((24,T),101,dtype=torch.long); Y=torch.full((24,T),-100,dtype=torch.long)
    for i,(seq,ans) in enumerate(items):
        X[i,:len(seq)]=torch.tensor(seq); Y[i,len(seq)-1]=ans
    opt.zero_grad()
    logits=m(X)
    loss=nn.CrossEntropyLoss(ignore_index=-100)(logits.reshape(-1,VOCAB),Y.reshape(-1))
    loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
    if (step+1)%1000==0:
        print(f'step {step+1} loss {loss.item():.3f} acc_u {acc("uniform"):.3f}',flush=True)
print('FINAL 1000-sample eval:')
for qm in ['uniform','first','mid','tail','uniform']:
    print(f'  {qm}: {acc(qm,1000):.3f}',flush=True)
