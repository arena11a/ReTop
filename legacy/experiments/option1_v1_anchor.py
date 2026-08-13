import sys, torch, random
import os
sys.path.insert(0, os.path.join(*([os.path.dirname(os.path.abspath(__file__))] + ['..']*2)))
import torch.nn as nn, torch.nn.functional as F
from hmn_v2 import HelixCouplingBlock, DifferentiableEpisodicMemory, ReversibleFunction
torch.manual_seed(0); rng=random.Random(0)
VOCAB=120; KEY_RANGE=(0,50); VAL_RANGE=(50,100); MARKER=100
def mk(N=8):
    keys=rng.sample(range(*KEY_RANGE),N); vals=[rng.randint(*VAL_RANGE) for _ in range(N)]
    pairs=list(zip(keys,vals)); q=rng.randint(0,N-2)
    seq=[]
    for k,v in pairs: seq+=[k,v]
    seq.append(MARKER); seq.append(pairs[q][0])
    return seq,pairs[q][1]
# Option 1: backbone (SSM) handles sequence; memory branch reads RAW embeddings; combine at head
class Mod(nn.Module):
    def __init__(self,D=64,state=8,L=2,n_cells=32):
        super().__init__()
        self.e=nn.Embedding(VOCAB,D)
        self.blocks=nn.ModuleList([HelixCouplingBlock(D,state) for _ in range(L)])
        self.memory=DifferentiableEpisodicMemory(D,n_cells,4)
        self.head=nn.Linear(D*2,VOCAB,bias=False)
    def forward(self,x):
        h=self.e(x)
        b=h
        for blk in self.blocks:
            b=ReversibleFunction.apply(b,[blk])
        m=self.memory(h)          # memory on RAW embed (token identity preserved)
        z=torch.cat([b,m],dim=-1) # combine at head
        return self.head(z)
def run(N,steps=2500,tag=''):
    torch.manual_seed(0); rng2=random.Random(0)
    m=Mod(); opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
    def acc(n=300):
        m.eval(); ok=0
        with torch.no_grad():
            for _ in range(n):
                seq,ans=mk(N); ok+=(m(torch.tensor([seq]))[0,-1].argmax(-1).item()==ans)
        m.train(); return ok/n
    best=0
    for step in range(steps):
        items=[mk(N) for _ in range(24)]
        T=max(len(x) for x,_ in items)
        X=torch.full((24,T),101,dtype=torch.long); Y=torch.full((24,T),-100,dtype=torch.long)
        for i,(seq,ans) in enumerate(items):
            X[i,:len(seq)]=torch.tensor(seq); Y[i,len(seq)-1]=ans
        opt.zero_grad()
        logits=m(X)
        loss=nn.CrossEntropyLoss(ignore_index=-100)(logits.reshape(-1,VOCAB),Y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
        if (step+1)%500==0:
            a=acc(); best=max(best,a); print(f'[{tag}] step {step+1} loss {loss.item():.3f} acc {a:.3f}',flush=True)
    return best
print('=== Option 1 (raw-embed memory || backbone, cat at head), 8 pairs ===')
r=run(8,2500,'opt1_8p')
print(f'\nBEST: option1_8p={r:.3f}')
