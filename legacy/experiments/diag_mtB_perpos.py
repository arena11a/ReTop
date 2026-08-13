import sys, torch, random
import os
sys.path.insert(0, os.path.join(*([os.path.dirname(os.path.abspath(__file__))] + ['..']*2)))
import torch.nn as nn, torch.nn.functional as F
from hmn_v2 import HelixCouplingBlock, ReversibleFunction
VOCAB=120; N=8; KEY=(0,50); TENS=(50,60); ONES=(60,70); MARKER=100

def mk(rng, q):
    keys=rng.sample(range(*KEY),N)
    vals=[rng.randint(0,99) for _ in range(N)]
    pairs=list(zip(keys,vals))
    seq=[]
    for k,v in pairs:
        seq += [k, TENS[0]+v//10, ONES[0]+v%10]
    seq += [MARKER, pairs[q][0]]
    tq=TENS[0]+pairs[q][1]//10; oq=ONES[0]+pairs[q][1]%10
    return seq, tq, oq

class MemVar(nn.Module):
    def __init__(self, dim, n_cells, top_k, beta_init=30.0, usage_decay=True, combined=False):
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
        self.combined=combined
        if combined:
            self.comb_val=nn.Linear(dim*2,dim)
    def forward(self,x):
        B,T,D=x.shape
        keys=self.cell_keys.unsqueeze(0).expand(B,-1,-1)
        values=self.cell_values.unsqueeze(0).expand(B,-1,-1)
        beta=self.beta.abs()+1.0
        usage=torch.zeros(B,self.n_cells,device=x.device)
        reads=[]; prev_h=torch.zeros(B,D,device=x.device); prev_prev_h=torch.zeros(B,D,device=x.device)
        for t in range(T):
            h=x[:,t]
            q=self.read_proj(h)[...,:D]
            sim=(q.unsqueeze(1)@keys.transpose(-1,-2)).squeeze(1)/(D**0.5)
            w=(sim*beta).softmax(dim=-1)
            m=(w.unsqueeze(-1)*values).sum(dim=-2)
            gate=torch.sigmoid(self.gate_proj(torch.cat([h,m],dim=-1)))
            reads.append(gate*h+(1-gate)*m)
            v=self.val_proj(h)
            erase=torch.sigmoid(v[...,:D]); add=v[...,D:2*D]
            strength=torch.sigmoid(v[...,2*D:]).squeeze(-1)
            is_val_last = (self.combined and t>=2 and t%3==2 and t<2*N)
            if is_val_last:
                wk=F.normalize(self.key_proj(prev_prev_h),dim=-1)
                combined_val=self.comb_val(torch.cat([prev_h,h],dim=-1))
            else:
                wk=F.normalize(self.key_proj(prev_h),dim=-1)
                combined_val=None
            wsim=(wk.unsqueeze(1)@keys.transpose(-1,-2)).squeeze(1)/(D**0.5)
            write_w=(wsim*beta).softmax(dim=-1)*strength.unsqueeze(-1)
            if self.usage_decay:
                write_w=write_w*(1-usage)
            if is_val_last:
                values=values*(1-write_w.unsqueeze(-1))+write_w.unsqueeze(-1)*combined_val.unsqueeze(1)
            else:
                keys=keys*(1-write_w.unsqueeze(-1))+wk.unsqueeze(1)*write_w.unsqueeze(-1)
                values=values*(1-write_w.unsqueeze(-1)*erase.unsqueeze(1))+write_w.unsqueeze(-1)*add.unsqueeze(1)
            usage=usage+write_w
            prev_prev_h=prev_h; prev_h=h
        return torch.stack(reads,1)

class Mod(nn.Module):
    def __init__(self,D=64,state=8,L=2,n_cells=256,design='B',**memkw):
        super().__init__()
        self.design=design
        self.e=nn.Embedding(VOCAB,D)
        self.blocks=nn.ModuleList([HelixCouplingBlock(D,state) for _ in range(L)])
        self.memory=MemVar(D,n_cells,4,combined=(design=='B'),**memkw)
        self.gate=nn.Linear(D*2,D)
        if design=='A': self.head=nn.Linear(D,VOCAB,bias=False)
        else:
            self.head_t=nn.Linear(D,10,bias=False)
            self.head_o=nn.Linear(D,10,bias=False)
    def forward(self,x):
        h=self.e(x)
        b=h
        for blk in self.blocks:
            b=ReversibleFunction.apply(b,[blk])
        m=self.memory(h)
        g=torch.sigmoid(self.gate(torch.cat([b,m],dim=-1)))
        z=g*b+(1-g)*m
        if self.design=='A': return self.head(z)
        return self.head_t(z), self.head_o(z)

def eval_per_pos(m, n=300):
    rng=random.Random(999)
    m.eval()
    with torch.no_grad():
        for q in range(N-1):   # q in 0..6
            ok=0; ok_t=0; ok_o=0
            for _ in range(n):
                seq,tq,oq=mk(rng,q)
                lt,lo=m(torch.tensor([seq]))
                lt=lt[0,-1]; lo=lo[0,-1]
                tp=TENS[0]+lt.argmax().item(); op=ONES[0]+lo.argmax().item()
                ok+=(tp==tq and op==oq); ok_t+=(tp==tq); ok_o+=(op==oq)
            print(f'  q={q}: full={ok/n:.3f} tens={ok_t/n:.3f} ones={ok_o/n:.3f}',flush=True)
    m.train()

torch.manual_seed(0); rng=random.Random(0)
m=Mod(design='B'); opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
for step in range(3000):
    items=[]
    for _ in range(24):
        seq,tq,oq=mk(rng,random.randint(0,N-2)); items.append((seq,tq,oq))
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
    if (step+1)%1000==0:
        print(f'step {step+1} loss {loss.item():.3f}',flush=True)
print('FINAL per-position eval (q=0..6):',flush=True)
eval_per_pos(m)
