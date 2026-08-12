# HMN v3 — Helix Register-Network
### สถาปัตยกรรม AI ที่ "เหนือกว่า" AI สุ่ม-ผ่านชั้นเดียว (single-pass LLM) : คัดลอกได้แม่น, คิดได้ลึก, ประหยัดหน่วยความจำ

---

## 0. สรุปเหตุผลว่าทำไมต้องออกแบบใหม่ (จากหลักฐานจริงของโปรเจกต์)

ทุกสถาปัตยกรรมที่เคย tested กับสร้างเป็น single-pass probabilistic generator:
"อ่าน context → forward หนึ่งรอบ → softmax → sample" แปลว่า:

| ข้อจำกัดของ AI แบบเดิม (verified ในโปรเจกต์เอง) | หลักฐาน |
|---|---|
| contextualization (SSM/MoE) ทำลาย content-address ของหน่วยความจำ | HMN เต็ม: 14% → memory อ่าน raw embedding: 62% → +β30+usage: 99% (task2_findings #3,#10) |
| slot-copy เป็นเพดาน — LM head ออก soft distribution ไม่มีวิธีก๊อบปี้ token จาก prompt เป๊ะ | val exact 0/40, train exact 9/40, pip slot ผิด 100% val (distill_design) |
| คิดหลายขั้น vs คิดครั้งเดียว ต้องนับขั้นเป็น tokens เพิ่ม (chain-of-thought) — เสีย tokens, ชา, ควบคุมยาก | distillation ต้องการ "rule + copy + verify" ครบ แต่ single-pass ทำไม่ได้ |
| ไม่มีการยืนยันคำตอบ (self-check) ใน forward เดียว | run_exact / syntax ยังไม่ 100% แม้ rule ถูก |

**สมมติฐานหลักของ v3:** คำตอบที่ถูกต้องมีสองชนิดที่ต้องการช่องสัญญาณ (channel) ต่างกัน
- **Literal / copy** (ค่าที่มีอยู่แล้วใน context: "pip install {pkg}", เลขจากหน่วยความจำ) → ต้องมีช่องก๊อบปี้ตรง (hard copy) ไม่ใช่ดูดออกจาก softmax
- **Computed / generated** (กติกาใหม่, paraphrase, reasoning) → ต้องมีช่องประมวลผล (abstraction) ซึ่ง v2 ทำได้ดีอยู่แล้ว

AI ที่เหนือกว่าคือ AI ที่รู้ว่า **เมื่อไหร่ต้อง copy เมื่อไหร่ต้อง generate และจัดจ้างเวลา "คิด" เพิ่มได้เมื่อยาก** — ไม่ใช่แค่ soft attention เดียวที่ทำให้ทุกอย่างคลุมเครือ

---

## 1. สามแนวคิดหลักของ v3 (novel contributions)

### 1.1 Two-Register Processing (WR + IR)

- **Working Register (WR)** — backbone เดิม (reversible SSM + MoE): สร้าง contextual abstraction ใช้ reasoning
- **Identity Register (IR)** — ตาราง slot ใหม่ (global, content-addressable, read only) เก็บ **token identity ดิบ**
  ไม่ถูก contextualize โดย SSM/MoE → สรุป: หลักฐาน #3 พิสูจน์แล้วว่าการทำแบบนี้ให้ recall **99%**

### 1.2 Dual-Head Decoder: Copy-Head ⊕ Generate-Head

Decoder ไม่มี head เดียว desเท่านั้น แต่มีสองหัว แบ่งงานตาม quantum ของคำตอบ:

- **Copy-Head** (head ที่เขียนด้วย `nn.Linear` → logit ของ token *จาก IR ที่ retrieve ตรง*):
  อ่าน memory ตำแหน่งที่ WR ระบุ address → เอาค่า token **ตรงๆ** เป็น candidate
- **Generate-Head** (LM head เดิม): softmax เหนือ vocab
- **Router กลาง**: `g = σ(W·[h, mem_read])` — ถ้า copy-head มี **confidence สูง** (การ gating ฝั่ง memory)
  → ตอบด้วย hard copy (exact); ถ้าไม่ → generate

> ต่างจาก pointer-generator (pointing เข้า attention weights) คือ copy-head ของเราชี้ไปยัง **episodic memory
> map ทุก cell** (ไม่ใช่แค่ input window) — จึงก๊อบปี้ได้ทั้ง token ที่อยู่ใน context หลายพัน tokens ก่อนหน้า
> และค่าที่ memory รวมเขียนเก็บไว้ (multi-token value) → สองสิ่งมาถึง focus เสมอโดยไม่หาย

### 1.3 Adaptive Latent Thinking (Latent CoT)

ชดเชยข้อจำกัด "single-pass" ที่ AI ทั่วไปมี: หลัง WR forward รอบแรก ไม่ output ทันที แต่มี
**deliberation loop** — รัน recurrence ต่อใน latent space (ไม่ใช่ token) K รอบ แล้วหยุดเมื่อ convergence:

```
for k in range(K_max):            # K adaptive
    h_k = block_clip(h_{k-1})     # refine hidden โดยไม่ต้อง decode เป็นคำ
    conf = output_confidence(h_k) # entropy ของ head ล่าง
    if conf ถึงเกณฑ์ -> break
logits = dual_head(h_k)           # ถ่ายทอดรอบสุดท้าย
```

- **คิด** โดยไม่เปลือง tokens, ไม่ใช่ chain-of-thought แบบข้อความ
- คุ้มบน CPU เพราะเป็นการรัน backbone ซ้ำ (reversible → activation ถูก reconstruct ฟรี)
- **K ปรับตามความยาก** ของ input (entropy-based) — แก้ค่า use-compute ตามปัญหา ได้ AI-"ใช้เวลา/คิดเพิ่มได้"

---

## 2. สถาปัตยกรรมรวม (Full Assembly)

```
input_ids
   │
   ├── ▶ [Embedding] ──────────────┬──────▶ Identity Register (IR)
   │                               │        ("literal lane", ใช้ token ดิบ)
   │                               │
   │   [WR: L× (Reversible SSM ⊕ MoE)]
   │                 │
   │                 └── [IR unify]: address ← WR h_t, retrieve IR → mem_read
   │
   │   [Thinking Buffer]   loop K: WR re-run on last hidden + gated mem_read
   │
   └── [Dual Head]
          ├── Copy-Head   : memory read -> exact candidate (hard)
          └── Generate-Head: LM softmax
          → gate g:  g·copy_logits + (1-g)·gen_logits   (g เรียนรู้ว่าเมื่อไหร่ copy)
```

IR เป็นช่องที่อ่านข้อมูลดิบทุก forward (ไม่ใช่แค่ตอนมี instruction) เพื่อให้ Copy-Head มีที่ก๊อบปี้ได้เสมอ

---

## 3. เกณฑ์ success ที่วัดได้ (ตรงเพดานที่พังของ v2)

| Task | v2 (verified) | v3 เป้าหมาย |
|---|---|---|
| Slot-copy: `pip install {pkg}` บน 120 val (ไม่เห็น slot ใน train) | 0/40 exact, pkg ผิดเกือบหมด | **≥90% copy-exact** (ผ่าน hard-copy) |
| Recall single-token (8 คู่/50 keys, random-query) | 97-99% | ไม่แย่ลง (≥97%) |
| syntax/API บน unseen templates | syntax 70%, API 89% | ≥ v2 + run_exact สูงขึ้น |
| งาน composite "copy+compute" (e.g. บวกค่าที่ต้อง copy ออกจาก memory ก่อน) | ยังไม่ได้ทดสอบ | ≥80% |

---

## 4. แผน implementation + validation (CPU-friendly)

1. **Write `IdentityRegister`** — ตาราง slot (cells) เก็บ `(key_id, value_ids[])`, content-address ด้วย raw embed,
   read = hard-attention (retrieve เต็ม value) — แยกจาก memory ของ v2 (ไม่ต้องการ training gate ซับซ้อน)
2. **Write `DualHead`** — copy-head อ่านจาก IR + gate; backward ที่เป็น trainable soft blend
   โดย forward ตอบ hard copy เมื่อ gate สูง (STE เดิม) — เหมือนแนวคิด v2.3 STE
3. **Write `LatentBuffer`** — deliberation loop, greedy best-K (K adaptive จาก entropy)
4. **Test harness `experiments/poc_hmn_v3.py`** — 3 tasks ข้างต้น + เทียบ v2

ทุกขั้น train ได้บน CPU (~0.2-0.4 s/step) ดังนั้นไม่จำกัดเวลาในการคิด/ทดสอบตามที่ต้องการ

---

## 5. สิ่งที่ยืนยันแล้วจาก v2 → ใช้ต่อใน v3 (ไม่ทดสอบซ้ำ)

- Pre-LN บังคับ (Post-LN ระเบิด step 63) — v2 doc §18 ✅
- Reversible backbone reconstruct แม่นยำ (error ~1e-7) — ประหยัด activation สำหรับ thinking loop ✅
- β=30 + usage_decay + key/value split ทำให้ memory recall 99% ✅
- Combined write (multi-token value ใน cell เดียว) ✅
- STE สำหรับ continuous gate (soft in backward, hard in forward) ✅

---

## 6. สิ่งที่ยังเป็น open question ของ v3 (วางไว้ให้ทดสอบ)

- IR ควรเป็น "cell จริง" หรือ "pointer ไป input"? (แล้วแต่ tradeoff: cell = สากลกว่า, pointer = ประหยัดกว่า)
- copy-head ควร integrate กับ memory cell (เขียน+อ่าน) หรือแยก register สำหรับ literal copy เท่านั้น
- latent thinking ควรวนบน hidden ของ layer ใด (top vs middle) จึง maximized convergence
- gate ควรเป็น per-token สำหรับ sequence (บาง token copy บาง token generate) — จุดนี้ยิ่งทำให้ strong

---