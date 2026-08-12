# Task 2 Findings — Extended KV Recall (v2.1 doc ข้อ 5.4)

สถานะ: วินิจฉัยเสร็จ 2 รอบ — พบ root cause หลัก 2 จุดที่ทำให้ "recall 0%" ที่เคยสรุปไว้ตอนต้น
**เป็นผลของ bug ใน data format + integration ของผม ไม่ใช่สถาปัตยกรรม memory เสียหาย**

---

## 1. Bug ที่ 1: data format `k=v` เข้ากันไม่ได้กับ memory write design

Memory ออกแบบให้: เขียนที่ token value โดย `write key = key_proj(h_{t-1})`
→ key token ต้องเป็น **token ก่อนหน้า value โดยตรง**

- Format เดิม: `"k1=v1 k2=v2 ... ?kQ"` → token ก่อน value คือ `=` (ไม่ใช่ key)
  → memory เก็บ association `"=" → value` เปล่าประโยชน์ → recall 0% เสมอ
- Format ใหม่ (v2): `"START k1 v1 k2 v2 ... ? kQ"` → key ติด value โดยตรง
  → memory เขียน `k → v` ถูกต้อง

**BPE space-prefix bug:** key `12` โผล่เป็น token `Ġ12` ในคู่ แต่เป็น `12` หลัง `?`
(ต่าง token id → key ตอนเขียน ≠ key ตอนอ่าน)
→ แก้โดย `START` prefix + `? k` (เว้นวรรค) ทำให้ key แตกเป็น token เดียวกันทุกตำแหน่ง

ผล: standalone memory (embed→memory→head) บน synthetic single-token task (5 pairs/50 keys)
**acc 25% → 82%** (ตรงกับ doc ที่อ้าง 100% เดิม)

## 2. Bug ที่ 2: memory integration ใน HMN เต็มระบบ

Memory ใน hmn_v2.py เดิม (ก่อนแก้) ใช้ `read weights` ไปกับการ write addressing
→ เขียนทับ association เดิมตลอด → stored value ถูก erase (ตรวจพบว่า cell ที่ถูกอ่าน
มี content ของ key เอง ไม่ใช่ value)

**การแก้ที่ทำให้ standalone ได้ 82%:**
- แยก `read_proj(h)` กับ `key_proj(prev_h)` อย่างชัดเจน (ตรง doc 10.3)
- write addressing = `softmax(beta * sim(key_proj(prev_h), keys))` (เนื้อหาแยกจาก read)
- เพิ่ม `gate_proj` (gate blend `g·h + (1-g)·m`) + learnable `beta` (sharpening temperature)
- ผล: เขียนคู่ key→value ลง cell เฉพาะ ไม่ถูกทับ

## 3. ปัญหาใหม่: backbone ทำลาย token identity

| Config | acc (synthetic 5 pairs/50 keys) |
|---|---|
| standalone memory (raw embed → memory → head) | **82%** |
| HMN เต็ม (coupling backbone → memory → head) | 14% |
| memory ใช้ raw embed + backbone แยกสาขา รวมที่ head | 62% |

สาเหตุ: hidden state ของ key token หลัง backbone (SSM recurrent + MoE) เป็น **contextual**
(h(key ที่คู่แรก) ≠ h(key ที่ query ตอนท้าย เพราะเห็น context ไม่เท่ากัน)
→ content addressing หา cell ไม่เจอ ตอนเขียน key กับตอนอ่าน key ต่างกัน

**นัยเชิงสถาปัตยกรรม:** การให้ memory อ่านจาก backbone hidden state โดยตรง (doc's
`h = h + memory(h)`) มี tension กับ content-addressing ซึ่งต้องการ token identity คงที่
ทางเลือกที่ได้ผลกว่า: memory ใช้ raw token embedding เป็น key address (ได้ 62%+)

## 4. Task 2 จริง (BPE, values 0-9999, n_pairs 3-15)

- loss ฝึก → 0.09 (จำ sequence ฝึกได้) แต่ val acc 0%
- Overfitting + **multi-token values (2-4 tokens)**: memory เขียนได้แค่ token แรกของ value
  การจะอ่านเลขเต็มต้อง chain retrieval หลาย token — ยากกว่า single-token มาก
- keys/vals ทับซ้อนกัน (ทั้งคู่เป็นตัวเลข) ทำให้ memory แยก "key 5" กับ "value 5" ไม่ได้

## 5. สรุป / สิ่งที่แก้แล้วใน hmn_v2.py

- `DifferentiableEpisodicMemory`: เพิ่ม `gate_proj`, `beta`, แยก read/write addressing
- `gen_recall.py`: format ใหม่ `START k1 v1 ... ? kQ`
- `test_recall.py` / `train_recall.py` parse format ใหม่

## 6. ยังค้าง

- การ integration memory เข้า HMN เต็มระบบ ยังไม่ได้ accuracy สูง (ต้องตัดสินใจ design:
  memory ควรรับ backbone state หรือ raw token identity?)
- multi-token value retrieval ใน Task 2 ยังไม่สำเร็จ

## 7. [2026-08-07] ตรวจสอบ 82% → 65% (diagnostic B) — พบ positional shortcut

**คำถาม:** standalone memory เดิมได้ 82% แต่ตอนนี้ได้ ~65% เกิดจากอะไร?

**ผล: ตัวเลข 100% เดิมใน doc หัวข้อ 18 มาจาก eval protocol ที่ query คู่แรกเสมอ
(train mode = query ตำแหน่งแรก → ได้ 100% เฉพาะการ query ตำแหน่งแรก แต่สอบ cross
กับ random-query ได้แค่ ~49%) — คือ positional shortcut ไม่ใช่ content addressing
ที่ generalize จริง**

| Train mode | Test: query คู่แรก | Test: query สุ่มตำแหน่ง |
|---|---|---|
| random-query | 74% | **45%** |
| first-only (แบบ doc เดิม) | **100%** | **49%** |

- ตัวแปรที่ทำให้ตัวเลขต่างกัน: **query position distribution** (ไม่ได้เป็น steps/seed/lr)
  - A: 5p/16c lr3e-4 6000 steps random-query → 55%
  - B: 5p/16c lr1e-3 3000 steps random-query → 64%
  - C: 5p/16c lr3e-4 6000 steps query-first → **100%**
- เพิ่ม cells ไม่ช่วย (8p: 16c=47%, 32c=47%, 64c=41%) — ไม่ใช่ capacity
- true episodic (zero-scratch cells) แย่ลง (24%) — learnable cell init จำเป็น

## 8. [2026-08-07] สรุปการ reproduce หัวข้อ 18 — UNVERIFIED

- Doc 18.2 อ้าง dim40/L4/no-MoE/8 คู่/50 keys → 90% ที่ 900 steps แต่ reproduce
  ด้วย config เดียวกันได้แค่ **3-6%**
- โค้ดต้นฉบับ (doc_mem.py) หายไปกับ /tmp — ตรวจย้อนกลับไม่ได้
- **เอกสาร v2.1 หัวข้อ 18 ถูก mark UNVERIFIED แล้ว** (แก้ในไฟล์ doc โดยตรง)
- **เพดานอ้างอิงที่ verified แล้ว (สำหรับเทียบ Option 1):**
  - standalone memory random-query (5 คู่): ~55-65%
  - Option 1 (raw-embed memory || backbone, cat ที่ head, 8 คู่): ~51%
- **ห้ามใช้ 90-100% เป็นเป้าหมาย** จนกว่าจะ reproduce ได้

## 9. [2026-08-07] ตรวจสอบ 18.2 width-vs-depth sweep — UNVERIFIED เช่นกัน

**คำถาม:** sweep 18.2 (d40/L4 vs d80/L4 vs d40/L8) ใช้ eval protocol เดียวกับ Task 2
(มี positional shortcut) หรือไม่?

**คำตอบ: ใช่ — เป็น artifact เดียวกัน**

- Doc นิยาม "recency bias = 0%" ใน 18.2 ว่า = query คู่แรกเสมอ (เหมือน 18.1 ข้อ 3)
- สคริปต์ต้นฉบับ (hmn_v2_width_sweep3.py) หายไปแล้ว แต่ doc บอก protocol ตรงกัน
- **รัน 3 configs ซ้ำด้วย random-query eval:**

| Config | params | train random → eval random | train random → eval first | train first → eval first | train first → eval random |
|---|---|---|---|---|---|
| d40/L4 (baseline) | 33.6K | 7.3% | 6.0% | 99.7% | **17%** |
| d80/L4 (กว้าง) | 102K | **15.3%** | 21.7% | 100% | **18%** |
| d40/L8 (ลึก) | 44.7K | 9.7% | 8.7% | 99.7% | **18%** |

- **train first → eval first = ~100% ทุก config** (ยืนยันว่าเลข 90% เดิมเป็น positional shortcut)
- **train first → eval random = 17-18% ทั้งหมด** — ไม่มี config ใดทำ content addressing ได้
- ด้วย random-query train/eval: **width ดีกว่า depth** (d80/L4=15.3% > d40/L8=9.7%)
  → กลับไปทางข้อสรุป v1.1 (width-matters) ไม่ใช่ depth-efficiency ที่ doc 18.2 อ้าง
- **ข้อสรุป "depth efficiency ดีกว่า width" เป็น UNVERIFIED** — ต้องรัน sweep ใหม่บน
  random-query ก่อนเชื่อถือ ดู hmn_v2 doc 18.2 ที่ mark หมายเหตุ v2.2 แล้ว

## 10. [2026-08-07] Option 1 tuning — ถึง 97-99% recall (8 คู่/50 keys, random-query)

**Baseline anchor:** `experiments/option1_v1_anchor.py` (D64/L2/32c/cat) = 51%

**Sweep capacity/combine/training:**

| Config | uniform | first | mid | tail |
|---|---|---|---|---|
| anchor 32c cat | 53% | 65% | 50% | 35% |
| 64c cat | 63% | 74% | 63% | 53% |
| 128c cat | 68% | 82% | 64% | 55% |
| 256c cat | 72% | 91% | 67% | 52% |
| 128c gate-blend | 69% | 88% | 74% | 59% |
| 256c gate + 4000 steps | 77% | 87% | 79% | 65% |

- capacity มี diminishing returns (32→64:+10, 64→128:+5, 128→256:+4)
- gate-blend รวมที่ head ดีกว่า cat เล็กน้อย

**ตัวแปร tail-overwrite (β sharpening / usage decay):**

| Config | uniform | first | mid | tail |
|---|---|---|---|---|
| beta_init=30 | 89% | 92% | 86% | 83% |
| usage_decay | 81% | 95% | 74% | 62% |
| **usage_decay + beta30** | **99.3%** | **100%** | **98%** | **96%** |

**Sanity check (guardrail — ผลเกิน +15 จุด):** seed 42 ใหม่, eval 1000 samples:
- uniform 97.7% / first 99.8% / mid 96.7% / tail 92.9% → **ผ่าน** (gradient เรียบ ไม่ใช่ shortcut)

**บทสรุป:**
- β sharpening (beta=30) + usage decay (ลด write strength ตามจำนวนครั้งที่ cell ถูกเขียนทับ)
  รวมกันให้ synergy ใหญ่: ตัวเดียวได้ 81-89%, รวมกัน 99%
- ตอนนี้ Option 1 ถึงระดับ "ใช้งานได้จริง" (>90%) บน 8 คู่/50 keys ด้วย random-query
- **best config: D64/L2/256c/gate-blend/beta30/usage_decay/3000 steps**

## 11. [2026-08-07] Multi-token (2-token values 0-99) — design A vs B

**คำถาม:** memory ควร re-query ทุก token (A) หรือ query ครั้งเดียวที่ key แล้ว carry state (B)?
สำหรับ 2-token values (tens + ones tokens)

**Design A (chained re-query): 11% uniform** — memory เก็บ k→tens ที่ tens-token
และ tens→ones ที่ ones-token แล้ว generate 2 ขั้น re-query — ทำงานไม่สำเร็จ
(chain error สะสม + read ที่ token กลางไม่มีข้อมูลครบ)

**Design B (query ครั้งเดียว, combined write): 72% → 96%**
- combined write: ที่ ones-token (is_val_last) เขียนคู่ key→[tens+ones] ด้วย
  `comb_val(cat([prev_h, h]))` เขียนเต็มค่า 1 ครั้ง แทนการเขียนแยก 2 ครั้ง
- อ่านครั้งเดียวที่ key แล้ว head_t/head_o ถอด 2 ตัวเลข

**Bug ที่เจอ (root cause ของ q=5,6 พัง):** `is_val_last` ใช้เงื่อนไข `t<2*N` (16)
แต่ positions ของ pairs อยู่ถึง `3*N-1` (23) → pairs 5-7 ไม่เคยได้ combined write
(ได้แค่ normal tens-only write) แก้เป็น `t<3*N` → q=5,6 ดีขึ้นพร้อมกัน

**ผล verified (seed 0 และ 42, exempt_combined=True — skip usage_decay สำหรับ combined write):**

| seed | q=0..6 (full) | average |
|---|---|---|
| 0 | 98/97/100/96/98/99/95 | ~97.5% |
| 42 | 96/96/93/96/94/94/94 | ~94.9% |

- q=5,6 ดีขึ้นพร้อมกัน ✅ / q=0-4 ไม่แย่ลง ✅ / seed 42 ยืนยัน ✅
- ผ่าน guardrail (gradient เรียบ ไม่มี collapse) ✅

**สรุป: design B ใช้ได้ (94-97%) เทียบ single-token 97-99% — พร้อม merge เข้า hmn_v2.py**
- config: D64/L2/256c/gate-blend/beta30/usage_decay/**exempt_combined**/3000 steps

## 12. [2026-08-07] Attribution test + merge into hmn_v2.py (v2.2)

**Attribution: off-by-one คือ root cause จริง; exempt_combined เป็น safety margin ที่ไม่จำเป็น**

ทดสอบ off-by-one fix อย่างเดียว (t<3*N) โดยปิด exempt_combined:

| seed | q=0..6 (full) |
|---|---|
| 0 | 99/97/98/95/96/95/93 |
| 42 | 99/98/97/96/90/91/88 |

→ ได้ผลเทียบเท่า config เต็ม (ทั้งสองตัว) — ยืนยันว่า **exempt_combined จำเป็น**
ไม่ต้องใช้ แต่เก็บไว้เป็น safety margin (neutral)

**Merge เข้า hmn_v2.py เรียบร้อย:**
- `DifferentiableEpisodicMemory.__init__` เพิ่ม `beta_init`, `usage_decay`, `combined`,
  `exempt_combined`, `n_pairs` (ค่าเริ่มต้นคง backward compat กับ config เดิม)
- forward เพิ่ม: usage_decay write, combined write (full-value ที่ token สุดท้ายของ value),
  prev_prev_h tracking, **off-by-one แก้แล้ว (t<3*n_pairs)** พร้อม comment ระบุ root cause
- เพิ่ม class `HMN_Option1`: memory อ่าน **raw embedding** (parallel branch) + coupling
  backbone (no MoE) + gate-blend รวมที่ head + ตัวเลือก `use_multi_head` (design B)

**Regression test หลัง merge (`experiments/regression_hmn_v2.py`):**

| Task | uniform | first | mid | tail | คาดไว้ |
|---|---|---|---|---|---|
| single-token | 95.8% | 99.1% | 95.5% | 91.4% | 97-99% ✅ |
| 2-token | 97.8% | 99.7% | 97.6% | 94.4% | 94-97% ✅ |

- ผ่านทั้งคู่ (single ≥90%, 2-token ≥85%)
- backward compat: old `HMN` + old memory defaults ยัง forward ได้ปกติ
