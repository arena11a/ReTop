import sys, torch, random
import os
sys.path.insert(0, os.path.join(*([os.path.dirname(os.path.abspath(__file__))] + ['..']*2)))
import torch.nn as nn, torch.nn.functional as F
from hmn_v2 import HelixCouplingBlock, ReversibleFunction

# 2-token values: v in 0..99 -> tens token = 50+v//10 (tokens 50-59), ones token = 60+v%10 (60-69)
# keys single-token 0-49, MARKER=100
VOCAB=120; N=8; KEY=(0,50); TENS=(50,60); ONES=(60,70); MARKER=100

def mk(rng, qmode):
    keys=rng.sample(range(*KEY),N)
    vals=[rng.randint(0,100) for _ in range(N)]
    pairs=list(zip(keys,vals))
    if qmode=='uniform': q=rng.randint(0,N-2)
    elif qmode=='first': q=0
    elif qmode=='mid': q=N//2-1
    else: q=N-2
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
            is_val_last = (self.combined and t>=2 and t%3==2 and t<2*N)  # [k,t,o] repeated, skip MARKER/query
            if is_val_last:
                wk=F.normalize(self.key_proj(prev_prev_h),dim=-1)  # key = pair's key token
                combined_val=self.comb_val(torch.cat([prev_h,h],dim=-1))  # tens+ones info
            else:
                wk=F.normalize(self.key_proj(prev_h),dim=-1)
                combined_val=None
            wsim=(wk.unsqueeze(1)@keys.transpose(-1,-2)).squeeze(1)/(D**0.5)
            write_w=(wsim*beta).softmax(dim=-1)*strength.unsqueeze(-1)
            if self.usage_decay:
                write_w=write_w*(1-usage)
            if is_val_last:
                # full replace of key cell with combined value (erase~1)
                values=values*(1-write_w.unsqueeze(-1))+write_w.unsqueeze(-1)*combined_val.unsqueeze(1)
            else:
                keys=keys*(1-write_w.unsqueeze(-1))+wk.unsqueeze(1)*write_w.unsqueeze(-1)
                values=values*(1-write_w.unsqueeze(-1)*erase.unsqueeze(1))+write_w.unsqueeze(-1)*add.unsqueeze(1)
            usage=usage+write_w
            prev_prev_h=prev_h; prev_h=h
        return torch.stack(reads,1)

class Mod(nn.Module):
    def __init__(self,D=64,state=8,L=2,n_cells=256,design='A',**memkw):
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

def run(design,steps=3000,tag=''):
    torch.manual_seed(0); rng=random.Random(0)
    m=Mod(design=design); opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
    def acc(qmode,n=300):
        m.eval(); ok=0
        with torch.no_grad():
            for _ in range(n):
                seq,tq,oq=mk(rng,qmode)
                if design=='A':
                    lt=m(torch.tensor([seq]))[0,-1]
                    tp=lt[TENS[0]:TENS[1]].argmax()+TENS[0]
                    seq2=seq+[int(tp)]
                    lo=m(torch.tensor([seq2]))[0,-1]
                    op=lo[ONES[0]:ONES[1]].argmax()+ONES[0]
                else:
                    lt,lo=m(torch.tensor([seq]))
                    lt=lt[0,-1]; lo=lo[0,-1]
                    tp=lt[TENS[0]:TENS[1]].argmax()+TENS[0]
                    op=lo[ONES[0]:ONES[1]].argmax()+ONES[0]
                ok+=(int(tp)==tq and int(op)==oq)
        m.train(); return ok/n
    best=0
    for step in range(steps):
        items=[]
        for _ in range(24):
            seq,tq,oq=mk(rng,'uniform'); items.append((seq,tq,oq))
        T=max(len(x) for x,_,_ in items)
        X=torch.full((24,T),101,dtype=torch.long); Y=torch.full((24,T),-100,dtype=torch.long)
        KEYPOS=torch.zeros((24,T),dtype=torch.bool); YT=torch.full((24,T),-100,dtype=torch.long); YO=torch.full((24,T),-100,dtype=torch.long)
        for i,(seq,tq,oq) in enumerate(items):
            X[i,:len(seq)]=torch.tensor(seq)
            Y[i,:-1]=torch.tensor(seq[1:])     # next-token supervision
            for idx in range(len(seq)):
                if idx%3==0 and idx<2*N:       # pair key positions: supervise two-head read
                    KEYPOS[i,idx]=True; YT[i,idx]=seq[idx+1]; YO[i,idx]=seq[idx+2]
            if design=='A':
                Y[i,len(seq)-1]=tq             # query key -> predict tens
            else:
                Y[i,len(seq)-1]=-100           # no next-token at query; use two-head
                KEYPOS[i,len(seq)-1]=True; YT[i,len(seq)-1]=tq; YO[i,len(seq)-1]=oq
        opt.zero_grad()
        out=m(X)
        if design=='A':
            loss=nn.CrossEntropyLoss(ignore_index=-100)(out.reshape(-1,VOCAB),Y.reshape(-1))
        else:
            lt,lo=out
            loss1=nn.CrossEntropyLoss(ignore_index=-100)(lt.reshape(-1,10),YT.reshape(-1))
            loss2=nn.CrossEntropyLoss(ignore_index=-100)(lo.reshape(-1,10),YO.reshape(-1))
            loss=loss1+loss2
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
        if (step+1)%1000==0:
            a=acc('uniform'); best=max(best,a); print(f'[{tag}] step {step+1} loss {loss.item():.3f} acc_u {a:.3f}',flush=True)
    au=acc('uniform'); af=acc('first'); am=acc('mid'); at=acc('tail')
    print(f'[{tag}] BEST_u={best:.3f} final: uniform={au:.3f} first={af:.3f} mid={am:.3f} tail={at:.3f}',flush=True)
    return best,au,af,am,at

print('=== DESIGN A: re-query each token (chained) ===')
run('A',3000,'A')
print('=== DESIGN B: query once, carry state ===')
run('B',3000,'B')
