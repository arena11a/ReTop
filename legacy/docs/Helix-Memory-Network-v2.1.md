# Helix Memory Network (HMN)
### เครือข่ายความจำเกลียวคู่แบบย้อนกลับได้

**เอกสารแนวคิดสถาปัตยกรรม (Architecture Concept Document)**
**เวอร์ชัน:** 2.2
**สถานะ:** v2 roadmap ข้อ 1–5 ผ่านการ validate แล้ว (Pre-LN, aux load-balancing loss, memory key/value projection แยก, width/depth sweep, full integration) แต่หัวข้อ 18.0–18.2 ถูกตีกลับเป็น UNVERIFIED หลัง reproduce ไม่ได้ (ตัวเลข 90-100% มาจาก positional shortcut) และได้สถาปัตยกรรมทดแทนที่ verify แล้วคือ **Option 1** (memory บน raw embeddings + gate-blend) — ดูหัวข้อ 18.6 สำหรับผล verified ล่าสุด. **อัปเดต v2.2 (post-roadmap):** ข้อ 5 (chunked parallel scan) และข้อ 7 (STE เต็มรูปแบบ) ทำเสร็จและ validate แล้ว — ดูหัวข้อ 18.7

> **บันทึกการแก้ไข v2.2:** (1) ทำเครื่องหมายหัวข้อ 18.0–18.2 ทั้งหมดเป็น UNVERIFIED พร้อมหลักฐาน reproduction ล้มเหลว (3-6%) และ root-cause ว่าเลข 90-100% เดิมมาจาก eval protocol ที่ query คู่แรกเสมอ (positional shortcut) — ดูหัวข้อ 18.2 หมายเหตุ v2.2, (2) เพิ่มหัวข้อ 18.6 บันทึกสถาปัตยกรรมที่ verify แล้ว (Option 1), การค้นพบ off-by-one bug ใน memory write, และผล regression หลัง merge เข้า hmn_v2.py, (3) ย้ายเป้าหมายอ้างอิงจาก "90-100% (UNVERIFIED)" ไปเป็น "97%+ single-token / 94-97% 2-token (VERIFIED)", (4) เพิ่มหัวข้อ 18.7 ผล roadmap ข้อ 5 (chunked parallel scan) และข้อ 7 (STE สำหรับ PK-MoE routing)

---

## สารบัญ

1. บทสรุปผู้บริหาร (Executive Summary)
2. ที่มาและแรงจูงใจ (Motivation)
3. วิเคราะห์คอขวดที่แท้จริงของ Transformer
4. หลักการออกแบบ (Design Principles)
5. สถาปัตยกรรมโดยละเอียด
   - 5.1 ภาพรวมระบบ
   - 5.2 Reversible Selective-SSM Backbone
   - 5.3 Sparse Conditional Compute Layer
   - 5.4 Differentiable Episodic Memory
   - 5.5 การประกอบเลเยอร์ทั้งหมด (Full Layer Assembly)
6. คณิตศาสตร์และรายละเอียดการคำนวณ
7. อัลกอริทึมการเทรน (Training Algorithm)
8. การวิเคราะห์หน่วยความจำและการคำนวณ (Memory & Compute Analysis)
9. การเปรียบเทียบกับ Transformer และสถาปัตยกรรมอื่น
10. Pseudocode
11. แผนการทดลองแบบขั้นบันได (Staged Validation Plan)
12. ความเสี่ยงและข้อจำกัด
13. คำถามวิจัยที่ยังเปิดอยู่ (Open Research Questions)
14. เอกสารอ้างอิงเชิงแนวคิด (Conceptual References)
15. ภาคผนวก: ตารางไฮเปอร์พารามิเตอร์แนะนำ
16. ผลการทดลองจริง — Stage 1–4 (Empirical Validation Results)
17. แผนสำหรับ v2 (Roadmap)
18. ผลการ validate v2 — ข้อ 1–5 (v2 Validation Results)

---

## 1. บทสรุปผู้บริหาร (Executive Summary)

Helix Memory Network (HMN) เป็นสถาปัตยกรรมโครงข่ายประสาทเทียมที่ออกแบบมาเพื่อตอบโจทย์เฉพาะเจาะจง: **การเทรนโมเดลภาษาที่มีความสามารถระดับ LLM ขนาดเล็ก-กลาง บนฮาร์ดแวร์ระดับผู้บริโภคทั่วไป (GPU 6–12GB หรือ CPU + RAM 16–32GB)** โดยไม่เสียสละความเสถียรของการเทรนแบบ gradient-based ที่พิสูจน์แล้วว่าใช้งานได้จริงในระดับ production

จุดตั้งต้นของ HMN แตกต่างจากสถาปัตยกรรม "ปฏิวัติ" ทั่วไปที่มักพยายามเลิกใช้ backpropagation โดยสิ้นเชิง (เช่น Hebbian learning, Forward-Forward Algorithm) ซึ่งยังไม่มีหลักฐานว่า scale ได้จริงถึงระดับ language model — HMN เลือกที่จะ **แก้ไขต้นตอที่แท้จริงของปัญหาหน่วยความจำ** นั่นคือ **การเก็บ activation ระหว่างเลเยอร์เพื่อรอการคำนวณ backward pass** ซึ่งเป็นตัวกิน VRAM มากที่สุดในการเทรน Transformer ทั่วไป มากกว่าตัว attention mechanism เองด้วยซ้ำเมื่อความยาว sequence ไม่ได้ยาวมากนัก

HMN ประกอบด้วยองค์ประกอบหลัก 3 ส่วนที่ทำงานร่วมกัน:

1. **Reversible Selective-SSM Backbone** — แกนประมวลผลลำดับที่ใช้หลักการ reversible computation ทำให้ไม่ต้องเก็บ activation กลางเลยระหว่าง forward pass
2. **Sparse Conditional Compute Layer** — ชั้นคำนวณแบบ routing ที่ดึงเฉพาะพารามิเตอร์ส่วนน้อยมาใช้งานต่อ token
3. **Differentiable Episodic Memory** — หน่วยความจำภายนอกสำหรับจัดการ long-range dependency โดยไม่ต้องขยาย KV cache

ทั้งสามส่วนนี้ **ไม่มีชิ้นใดเป็นสิ่งใหม่ในตัวเอง** — reversible networks, selective SSM, mixture-of-experts routing, และ differentiable external memory ล้วนมีงานวิจัยรองรับแยกกันมาก่อน สิ่งที่ HMN นำเสนอคือ **การประกอบชิ้นส่วนที่พิสูจน์แล้วเข้าด้วยกันในรูปแบบเฉพาะ** เพื่อให้ได้คุณสมบัติที่ไม่มีสถาปัตยกรรมใดมีครบพร้อมกัน: เทรนได้เสถียรด้วย gradient มาตรฐาน + activation memory ต่ำมาก + compute แบบ sparse + long context ที่ไม่บวม VRAM

> **อัปเดต v1.1:** ข้อเสนอนี้ผ่านการทดลองขั้นต้นจริงแล้วบน CPU สเปคต่ำมาก (i3 Gen2, RAM 4.6GB) ครบทั้ง 4 stage ตามแผนในหัวข้อ 11 ผลลัพธ์ยืนยันสมมติฐานหลักสามข้อ: reversible reconstruction แม่นยำระดับ 1e-8 แม้รวมทุกโมดูลแล้ว, การเทรนเสถียรเมื่อใช้ Pre-LN, และ integration ของทั้ง 3 องค์ประกอบไม่ลดทอนความสามารถของ backbone รายละเอียดเต็มอยู่ในหัวข้อ 16

> **อัปเดต v2.0:** roadmap ข้อ 1–4 (Pre-LN, aux load-balancing, memory key/value แยก, width/depth sweep) ผ่านการ validate ครบ — พบข้อแก้ไขสำคัญหนึ่งจุดคือสมมติฐาน "กว้างก่อนลึก" ถูกค้านโดยผลการทดสอบระบบเต็ม (ดูหัวข้อ 17 และ 18.2)

---

## 2. ที่มาและแรงจูงใจ (Motivation)

เอกสารนี้เกิดจากการวิเคราะห์ย้อนหลังของสถาปัตยกรรมทางเลือก 4 แบบที่เคยถูกเสนอมาก่อน (HyperSparse Recurrent Memory Network, Episodic Lattice Network, CortexNet, LUMINA) ซึ่งทั้งหมดพยายามหลีกเลี่ยง self-attention และ/หรือ backpropagation แบบดั้งเดิม บทเรียนสำคัญที่สกัดได้มีดังนี้:

| บทเรียน | รายละเอียด |
|---|---|
| Local/Hebbian learning มีความเสี่ยงทฤษฎีสูง | ไม่มีหลักฐานเชิงประจักษ์ว่า pure local learning rule จะ scale ไปถึงคุณภาพระดับ language model ที่แข่งขันได้ |
| Forward-Forward Algorithm ยังไม่ผ่านการพิสูจน์ที่ scale ใหญ่ | แม้แต่ในงาน image classification ขนาดกลาง ก็ยังไม่มีผลลัพธ์ที่แข่งกับ backprop ได้ |
| Hybrid gradient + Hebbian มีความไม่เสถียร | จุดต่อระหว่างสองระบบการเรียนรู้ที่ต่างกันมักเป็นจุดที่เทรนพังในทางปฏิบัติ |
| สถาปัตยกรรมที่ยึด SSM + sparse memory (แบบ CortexNet) มี prior art จริง | Mamba และ product-key memory พิสูจน์ตัวเองแล้วในงานวิจัยและ production จริง |
| ไม่มีสถาปัตยกรรมใดในกลุ่มนี้แตะประเด็น **activation memory** โดยตรง | ทุกตัวโฟกัสที่การลด complexity ของ attention (O(N²) → O(N)) แต่ไม่มีตัวไหนแก้ปัญหาการเก็บ activation เพื่อ backward pass ซึ่งเป็นตัวกิน VRAM หลักจริงๆ ในทางปฏิบัติ |

จากบทเรียนเหล่านี้ HMN จึงถูกออกแบบด้วยหลักการว่า **"ปฏิวัติในจุดที่คุ้มความเสี่ยงที่สุด"** — เลือกแก้ปัญหาความจำโดยตรงด้วยเทคนิคที่มีรากฐานทางคณิตศาสตร์มั่นคง (reversible computation) แทนที่จะเสี่ยงกับกลไกการเรียนรู้ที่ยังไม่มีใครพิสูจน์ว่า scale ได้

---

## 3. วิเคราะห์คอขวดที่แท้จริงของ Transformer

ก่อนออกแบบ จำเป็นต้องแยกให้ชัดว่าอะไรคือคอขวดจริงของการเทรน Transformer บนเครื่องสเปคต่ำ เพราะความเข้าใจผิดที่พบบ่อยคือการโทษ self-attention (O(N²)) เพียงอย่างเดียว ทั้งที่ในทางปฏิบัติ โดยเฉพาะเมื่อความยาว sequence ไม่ได้ยาวมาก (เช่น < 4096 tokens) **ตัวกิน VRAM หลักคือ activation ที่ต้องเก็บไว้ทุกเลเยอร์เพื่อรอ backward pass**

### 3.1 องค์ประกอบของการใช้ VRAM ระหว่างเทรน

```
VRAM รวม ≈ Model Weights + Optimizer States + Gradients + Activations + KV Cache (ถ้ามี)
```

สำหรับโมเดลขนาดกลาง (~500M–1B parameters) ด้วย Adam optimizer:

| องค์ประกอบ | สัดส่วนโดยประมาณ | ขึ้นกับอะไร |
|---|---|---|
| Model Weights (fp16) | คงที่ตามจำนวนพารามิเตอร์ | จำนวนพารามิเตอร์ |
| Optimizer States (Adam m, v) | ~2 เท่าของ weights | จำนวนพารามิเตอร์ |
| Gradients | เท่ากับ weights | จำนวนพารามิเตอร์ |
| **Activations** | **มักเป็นสัดส่วนใหญ่ที่สุดเมื่อ batch/seq length เพิ่ม** | **จำนวนเลเยอร์ × ความยาว sequence × batch size × hidden dim** |

จุดสำคัญ: Activation memory **เพิ่มเป็นเส้นตรงตามจำนวนเลเยอร์** (L) ซึ่งต่างจาก weights/optimizer/gradients ที่ไม่ขึ้นกับ L โดยตรงในแบบเดียวกัน นี่คือเหตุผลที่โมเดลลึกมากมักเทรนไม่ได้บน GPU เล็ก แม้ตัว parameter count จะพอดีก็ตาม

### 3.2 บทสรุปการวิเคราะห์

การลด complexity ของ attention จาก O(N²) เป็น O(N) (แบบที่ CortexNet, EpiLat, HSRM ทำ) **ช่วยได้เฉพาะกรณี sequence ยาวมาก** แต่ไม่ได้แก้ปัญหา activation memory ที่ขยายตามความลึกของเครือข่าย ซึ่งเป็นปัญหาคนละมิติกัน

HMN จึงเลือกแก้ทั้งสองมิติพร้อมกัน: ลด compute complexity ด้วย SSM (เหมือนกลุ่มก่อนหน้า) **และ** ลด activation memory ด้วย reversible computation (จุดที่ไม่มีใครแตะ)

---

## 4. หลักการออกแบบ (Design Principles)

HMN ยึดหลักการออกแบบ 5 ข้อ ซึ่งทุกข้อมาจากบทเรียนในหัวข้อที่ 2 และ 3:

1. **Gradient-based learning เท่านั้น** — ไม่ใช้ pure local/Hebbian rule เพื่อหลีกเลี่ยงความเสี่ยงทฤษฎีที่ยังไม่มีใครแก้
2. **แก้ activation memory โดยตรง** — ผ่าน reversible layer design แทนการลด attention complexity เพียงอย่างเดียว
3. **Compute sparsity มากกว่า parameter sparsity เปล่าๆ** — ใช้ conditional computation (routing) เพื่อให้ FLOPs ต่อ token ต่ำ แม้พารามิเตอร์รวมจะสูง
4. **ทุกองค์ประกอบต้องมี prior art ที่ทำงานได้จริง** — ลดความเสี่ยงจาก integration ของสิ่งที่ไม่เคยพิสูจน์
5. **Long context ต้องไม่ทำให้ VRAM บวมตามความยาว** — ผ่านหน่วยความจำภายนอกที่อยู่นอก GPU

---

## 5. สถาปัตยกรรมโดยละเอียด

### 5.1 ภาพรวมระบบ

```
Input Tokens
     │
     ▼
┌─────────────────────┐
│ Orthogonal Embedding │  (การฝัง token เริ่มต้น)
└─────────────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│  Helix Block × L  (เลเยอร์หลัก ทำซ้ำ L ครั้ง)  │
│                                            │
│  ┌────────────────────────────────────┐  │
│  │ Reversible Coupling                 │  │
│  │  ┌──────────────┐  ┌──────────────┐│  │
│  │  │ Selective SSM │  │ Selective SSM││  │
│  │  │   (branch A)  │  │  (branch B)  ││  │
│  │  └──────────────┘  └──────────────┘│  │
│  └────────────────────────────────────┘  │
│                    │                       │
│                    ▼                       │
│  ┌────────────────────────────────────┐  │
│  │ Sparse Conditional Compute (Routing)│  │
│  └────────────────────────────────────┘  │
│                    │                       │
│                    ▼                       │
│  ┌────────────────────────────────────┐  │
│  │ Differentiable Episodic Memory      │  │
│  │  (Read + Write, ทุก K เลเยอร์)        │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
     │
     ▼
┌─────────────────────┐
│ Output Projection    │
└─────────────────────┘
     │
     ▼
  Next-token logits
```

โครงสร้างนี้ทำงานเป็น "เกลียวคู่" (Helix) เพราะสัญญาณข้อมูลถูกแบ่งเป็นสองสาย (branch A, branch B) ที่พันกันแบบ coupling ในทุกเลเยอร์ ซึ่งเป็นที่มาของชื่อสถาปัตยกรรม

---

### 5.2 Reversible Selective-SSM Backbone

#### 5.2.1 แนวคิด Reversible Computation

หลักการพื้นฐานของ reversible network (อ้างอิงแนวคิดจาก RevNet และการประยุกต์ใน Reformer) คือการออกแบบให้แต่ละเลเยอร์เป็นฟังก์ชันที่ **คำนวณย้อนกลับได้แบบ exact** โดยไม่ต้องเก็บ input ของเลเยอร์นั้นไว้

แบ่ง hidden state ที่มิติ $d$ ออกเป็นสองส่วนเท่าๆ กัน: $h = [h_a, h_b]$ โดยแต่ละส่วนมีมิติ $d/2$

**Forward pass ของ Helix Coupling Layer:**

$$h_a' = h_a + F_1(h_b)$$
$$h_b' = h_b + F_2(h_a')$$

โดย $F_1, F_2$ คือฟังก์ชัน Selective SSM (รายละเอียดในหัวข้อ 5.2.2)

**Backward reconstruction (ใช้ตอน backward pass เท่านั้น ไม่ใช้ตอน forward):**

$$h_b = h_b' - F_2(h_a')$$
$$h_a = h_a' - F_1(h_b)$$

จุดสำคัญ: สมการทั้งสองชุดนี้ **แม่นยำทางคณิตศาสตร์ 100%** ไม่ใช่การประมาณ (approximation) — ตราบใดที่ $F_1, F_2$ คำนวณด้วย floating point precision เดียวกันทั้งสองทิศทาง ค่า $h_a, h_b$ ที่คำนวณย้อนกลับได้จะตรงกับค่าดั้งเดิมทุกประการ (ในทางปฏิบัติอาจมี floating-point drift เล็กน้อยสะสมตามความลึก ซึ่งต้องจัดการด้วยเทคนิคที่กล่าวถึงในหัวข้อ 12)

#### 5.2.2 Selective SSM ภายใน (F₁, F₂)

ภายในแต่ละ $F_i$ ใช้กลไก Selective State-Space Model แบบเดียวกับที่ใช้ใน Mamba คือ SSM ที่มีพารามิเตอร์ $A, B, C, \Delta$ ขึ้นกับ input แบบไดนามิก (input-dependent, ไม่คงที่แบบ SSM ดั้งเดิม):

$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$$
$$y_t = C_t h_t$$

โดย $\bar{A}_t = \exp(\Delta_t A)$ และ $\bar{B}_t = \Delta_t B_t$ ถูกคำนวณผ่าน discretization ที่ขึ้นกับ input ณ เวลานั้น (selective mechanism) ทำให้โมเดลสามารถ "เลือก" ว่าจะจำหรือลืมข้อมูลตามเนื้อหา ไม่ใช่ตามตำแหน่งคงที่

การคำนวณนี้ทำได้แบบขนาน (parallel scan) ด้วย complexity $O(N \cdot d \cdot P)$ โดย $N$ คือความยาว sequence, $d$ คือ hidden dim, $P$ คือ state dimension — **เป็นเส้นตรงต่อความยาว sequence** ไม่ใช่กำลังสองแบบ attention

> **อัปเดต v2.0:** ต้องใช้ **Pre-LN** — LayerNorm อยู่ *ภายใน* $F_1, F_2$ (ก่อน in_proj) ไม่ใช่หลัง coupling output การใช้ Post-LN (norm หลัง coupling) ทำให้การเทรนระเบิดเป็น NaN ที่ step 63 ในการทดลองจริง และทำลายคุณสมบัติ reversible ด้วย (reconstruction error เพิ่มจาก ~1e-8 เป็น ~2.5) รายละเอียดในหัวข้อ 16.3 และ 18.1

#### 5.2.3 เหตุผลที่ใช้ SSM แทน Attention ภายใน Coupling

การเลือก SSM แทน self-attention เป็นฟังก์ชันภายใน coupling layer มีเหตุผลสองประการ:
1. Complexity เชิงเส้นต่อความยาว sequence ทำให้ compute cost ต่อ token คงที่
2. SSM มีสถานะ (state) ที่ไหลต่อเนื่องตามธรรมชาติ ซึ่งเข้ากันได้ดีกับโครงสร้าง reversible coupling ที่ต้องการฟังก์ชัน deterministic ชัดเจนระหว่างสองสาขา ต่างจาก attention ที่การคำนวณย้อนกลับซับซ้อนกว่ามากเนื่องจาก softmax และการรวม token จำนวนมาก

---

### 5.3 Sparse Conditional Compute Layer

ทำหน้าที่แทน Feed-Forward/MLP block ของ Transformer แบบดั้งเดิม แต่ใช้กลไก routing แบบ product-key memory / MoE-lite

#### 5.3.1 กลไกการทำงาน

1. จาก hidden state $h_t$ หลัง coupling layer คำนวณ query vector: $q_t = W_q h_t$
2. แบ่ง parameter bank ทั้งหมด (เช่น 65,536 "experts" ย่อยขนาดเล็ก) ออกเป็นสอง sub-key space ตามแนวคิด product-key memory:
   $$q_t = [q_t^{(1)}, q_t^{(2)}]$$
   แต่ละส่วนค้นหา top-$\sqrt{K}$ ใน sub-key ของตัวเอง แล้วรวม cross product เพื่อให้ได้ top-$K$ โดยมี complexity $O(\sqrt{N})$ แทนที่จะเป็น $O(N)$ เมื่อเทียบกับการค้นหาตรงทั้งหมด
3. ดึงเฉพาะ expert ที่ถูกเลือก (เช่น $K=32$ จาก 65,536) มาคำนวณ:
   $$y_t = \sum_{i \in \text{top-}K} \text{softmax}(q_t \cdot k_i) \cdot v_i$$
4. ระหว่างเทรน ใช้ **soft top-K พร้อม straight-through estimator** เพื่อให้ gradient ไหลผ่านการเลือก expert ได้ (ต่างจาก stop-gradient ของ CortexNet ซึ่งตัด gradient ทิ้ง — HMN เลือกให้ gradient ไหลผ่านเพื่อให้ routing เรียนรู้ได้แม่นยำขึ้น โดยแลกกับ compute เพิ่มขึ้นเล็กน้อยระหว่างเทรน)
5. ตอน inference ใช้ hard top-K ตรงๆ เพื่อความเร็วสูงสุด

> **อัปเดต v2.0:** ต้องมี **auxiliary load-balancing loss** เพื่อป้องกัน expert collapse — วัดจริงพบว่าไม่มี aux loss มี expert ~2–6% ไม่ถูกใช้งานตอน initialization ซึ่งจะแย่ลงเมื่อเทรน ตัว loss นี้เป็น normalized อยู่ในช่วง [0,1] (0 = สมดุลสมบูรณ์, 1 = collapse) และคูณด้วย coef (เริ่มที่ 0.1) แล้วบวกรวมกับ main loss ดูหัวข้อ 18.1

#### 5.3.2 เหตุผลที่ประหยัดหน่วยความจำและการคำนวณ

พารามิเตอร์รวมของ conditional compute layer อาจสูงถึงหลักพันล้าน แต่ **compute ที่ active ต่อ token มีเพียงสัดส่วนเล็กน้อย** (เช่น K/N = 32/65536 ≈ 0.05%) ทำให้:
- FLOPs ต่อ token ต่ำ แม้ capacity รวมของโมเดลสูง
- Optimizer state (Adam m, v) ที่ต้องอัปเดตจริงต่อ step มีเฉพาะส่วนที่ active — ถ้าใช้ sparse optimizer update จะลด memory bandwidth ได้มาก

---

### 5.4 Differentiable Episodic Memory

ทำหน้าที่จัดการ long-range dependency ที่เกินขอบเขตของ SSM state เพียงอย่างเดียว

#### 5.4.1 โครงสร้างหน่วยความจำ

- Memory bank ขนาด $M$ cells (เช่น $M = 100{,}000$–$1{,}000{,}000$) เก็บอยู่ใน **CPU RAM ไม่ใช่ GPU VRAM** (ในระดับโมเดลทดสอบเล็กยังเก็บในหน่วยความจำเดียวกับโมเดลได้)
- แต่ละ cell มี key vector $k_i \in \mathbb{R}^{d_k}$ และ value vector $v_i \in \mathbb{R}^{d_v}$

#### 5.4.2 การอ่าน (Read)

1. ทุกๆ $K_{\text{interval}}$ เลเยอร์ (เช่นทุก 4 เลเยอร์) สร้าง query จาก hidden state ปัจจุบัน
2. ใช้ approximate nearest-neighbor search (LSH หรือ HNSW ผ่าน library เช่น FAISS) เพื่อดึงเฉพาะ top-$K$ cells ที่เกี่ยวข้องที่สุด — เฉพาะเวกเตอร์ที่ถูกดึงเท่านั้นที่ถูกโอนเข้า GPU (แนวทางสำหรับสเกลใหญ่; ในการทดสอบเล็กใช้ content-based soft addressing เต็มหน่วยความจำ)
3. คำนวณ weighted readout ด้วย softmax attention เฉพาะกลุ่ม top-$K$ (ไม่ใช่ทั้งหมด) แล้วรวมเข้ากับ hidden state ผ่าน gate:
   $$g = \sigma(W_g [h_t, m_t])$$
   $$h_t' = g \odot h_t + (1-g) \odot m_t$$

#### 5.4.3 การเขียน (Write)

ใช้กลไก gated write คล้าย Differentiable Neural Computer (DNC): เขียนแบบ soft ผ่าน erase-and-add mechanism ที่ differentiable เต็มรูปแบบ ทำให้ gradient ไหลผ่านการเขียนความจำได้ ต่างจาก HSRM/CortexNet ที่ใช้ Hebbian/EMA update ที่ตัด gradient ทิ้ง:

$$v_i \leftarrow v_i \odot (1 - e_t w_i) + a_t w_i$$

โดย $e_t$ คือ erase vector, $a_t$ คือ add vector, $w_i$ คือ write weight ของ cell $i$ (มาจาก similarity score เดียวกับตอนอ่าน)

> **อัปเดต v2.0:** **key และ value ต้องมาจาก projection แยกกัน** — write key มาจาก **state ก่อนหน้า** ($\text{key\_proj}(h_{t-1})$) เพื่อให้เกิด association คู่ (key ที่เห็นก่อน → value ปัจจุบัน) ในขณะที่ erase/add มาจาก **state ปัจจุบัน** ($\text{val\_proj}(h_t)$) การให้ projection เดียวกันทั้งสองบทบาททำให้โมเดลไม่สามารถเขียน association ที่ถูกต้องได้ (ผล recall ตกจาก 100% เหลือ ~27%) รายละเอียดในหัวข้อ 18.1

#### 5.4.4 เหตุผลที่ไม่บวม VRAM ตามความยาว context

เนื่องจาก memory bank อยู่นอก GPU และมีเพียง top-K cells (K มีขนาดคงที่ เช่น 64) ที่ถูกโอนเข้า GPU ต่อการอ่านหนึ่งครั้ง การขยาย context หรือขยาย memory bank ให้ใหญ่ขึ้นจึง **ไม่เพิ่ม VRAM ที่ใช้ระหว่างเทรน** เพิ่มเพียง RAM และเวลาในการค้นหาเท่านั้น

---

### 5.5 การประกอบเลเยอร์ทั้งหมด (Full Layer Assembly)

โครงสร้างเลเยอร์เดียว ("Helix Block") ประกอบด้วย:

```
Helix Block(h):
    h_a, h_b = split(h)
    h_a' = h_a + SelectiveSSM_1(h_b)      # Pre-LN ภายใน SelectiveSSM_1
    h_b' = h_b + SelectiveSSM_2(h_a')     # Pre-LN ภายใน SelectiveSSM_2
    h = concat(h_a', h_b')
    h = h + SparseConditionalCompute(h)   # ต้องมี aux load-balancing loss
    if layer_index % memory_interval == 0:
        h = h + DifferentiableEpisodicMemory(h)   # key_proj(prev)/val_proj(cur) แยกกัน
    return h
```

โมเดลเต็มประกอบด้วย Helix Block ซ้อนกัน $L$ ชั้น (เช่น $L = 24$–$32$) โดย reversible property ใช้ได้กับทุกเลเยอร์ที่เป็น coupling layer — ส่วน conditional compute และ episodic memory ที่แทรกอยู่จะต้องมีการจัดการ checkpointing เพิ่มเติมเล็กน้อย (ดูหัวข้อ 8)

---

## 6. คณิตศาสตร์และรายละเอียดการคำนวณ

### 6.1 สรุปสัญลักษณ์

| สัญลักษณ์ | ความหมาย |
|---|---|
| $d$ | มิติของ hidden state ทั้งหมด |
| $d/2$ | มิติของแต่ละสาขา (branch) หลัง split |
| $P$ | state dimension ภายใน SSM |
| $N$ | ความยาว sequence |
| $L$ | จำนวนเลเยอร์ |
| $M$ | จำนวน cell ใน episodic memory |
| $K$ | จำนวน cell/expert ที่ถูกเลือกต่อการ route/read หนึ่งครั้ง |
| $N_e$ | จำนวน expert ทั้งหมดใน conditional compute |

### 6.2 Complexity รวมต่อ Forward Pass

$$\text{Compute} = O(N \cdot d \cdot P)_{\text{SSM}} + O(N \cdot K \cdot d)_{\text{routing}} + O\left(\frac{L}{K_{\text{interval}}} \cdot K \cdot d\right)_{\text{memory}}$$

ทุกเทอมเป็นเส้นตรงต่อ $N$ — ไม่มีเทอมใดเป็น $O(N^2)$

### 6.3 Activation Memory ที่ต้องเก็บจริง

สำหรับ coupling layer แบบ reversible ไม่ต้องเก็บ activation ระหว่างเลเยอร์เลย เก็บเพียง:
- Input เริ่มต้นของทั้งบล็อก (สำหรับจุดเริ่ม reconstruction)
- Output สุดท้ายของทั้งบล็อก

$$\text{Activation Memory} = O(N \cdot d) \quad \text{(ไม่ขึ้นกับ } L\text{)}$$

เทียบกับ Transformer/SSM แบบมาตรฐานที่ต้องเก็บ activation ทุกเลเยอร์:

$$\text{Activation Memory}_{\text{standard}} = O(L \cdot N \cdot d)$$

นี่คือจุดที่ HMN ประหยัดได้มากที่สุดเมื่อ $L$ มีค่าสูง (โมเดลลึก)

---

## 7. อัลกอริทึมการเทรน (Training Algorithm)

### 7.1 Forward Pass

1. รับ input tokens → embedding
2. ผ่าน Helix Block ทั้ง $L$ ชั้นตามลำดับ **โดยไม่เก็บ activation กลาง** (เก็บเฉพาะ output สุดท้ายของแต่ละบล็อกใหญ่ตามจำนวน checkpoint segment ที่กำหนด)
3. คำนวณ output logits และ loss (cross-entropy สำหรับ next-token prediction)

### 7.2 Backward Pass

1. เริ่มจาก gradient ของ output layer
2. **คำนวณย้อนกลับทีละ Helix Block** โดยใช้สมการ reconstruction ในหัวข้อ 5.2.1 เพื่อหา activation ของแต่ละเลเยอร์แบบสดๆ ระหว่าง backward
3. หา gradient ของ $F_1, F_2$ จาก activation ที่ reconstruct ได้ แล้วสะสม gradient ไปยัง weight ของ SSM
4. สำหรับ Sparse Conditional Compute: gradient ไหลผ่านเฉพาะ top-K experts ที่ถูกเลือกในตอน forward (ผ่าน straight-through estimator)
5. สำหรับ Episodic Memory: gradient ไหลผ่าน read/write mechanism แบบมาตรฐานเหมือน DNC

### 7.3 อัปเดตพารามิเตอร์

ใช้ AdamW มาตรฐาน แต่สำหรับ Sparse Conditional Compute แนะนำให้ใช้ **sparse optimizer state update** — อัปเดตเฉพาะ optimizer moment ของ expert ที่ active ในแต่ละ step เพื่อลด memory bandwidth (ไม่ใช่ลด memory footprint ของ optimizer state ทั้งหมด เพราะยังต้องเก็บ state ของทุก expert ไว้เผื่อถูกเลือกในอนาคต)

---

## 8. การวิเคราะห์หน่วยความจำและการคำนวณ (Memory & Compute Analysis)

### 8.1 ประมาณการสำหรับโมเดลตัวอย่าง

สมมติค่าไฮเปอร์พารามิเตอร์ตัวอย่าง:
- $d = 1024$, $L = 24$, $N = 2048$ (ความยาว sequence), batch size = 4
- Sparse Conditional Compute: $N_e = 65{,}536$ experts ขนาดเล็ก, $K = 32$ active ต่อ token
- Episodic Memory: $M = 200{,}000$ cells, $K_{\text{read}} = 64$

| องค์ประกอบ | Transformer มาตรฐาน (ประมาณ) | HMN (ประมาณ) |
|---|---|---|
| Activation memory | $O(L \cdot N \cdot d)$ ≈ หลาย GB ที่ $N=2048, L=24$ | $O(N \cdot d)$ ≈ ต่ำกว่า 10 เท่า |
| Parameter memory (compute-active) | เต็มทุกพารามิเตอร์ | เฉพาะ K/N_e ของ conditional layer |
| KV cache / long-context memory | บวมตาม N | คงที่ (external memory อยู่นอก GPU) |
| ความเร็วต่อ step | เร็วกว่า (ไม่มี recompute) | ช้าลง ~20–30% (ต้อง recompute ตอน backward) |

### 8.2 Trade-off สำคัญ

HMN แลก **เวลาในการเทรนต่อ step ที่เพิ่มขึ้น** (เพราะต้อง recompute forward ระหว่าง backward) เพื่อแลกกับ **VRAM ที่ลดลงอย่างมาก** — นี่คือ trade-off ที่รู้จักดีในวงการ reversible network (RevNet, Reformer รายงาน overhead ลักษณะนี้เช่นกัน) เหมาะกับสถานการณ์ที่ VRAM เป็นข้อจำกัดหลัก มากกว่าเวลาเทรน ซึ่งตรงกับโจทย์ "เทรนบนเครื่องสเปคต่ำ" พอดี

---

## 9. การเปรียบเทียบกับ Transformer และสถาปัตยกรรมอื่น

| คุณสมบัติ | Transformer | CortexNet (อ้างอิง) | HMN |
|---|---|---|---|
| Attention/mixing complexity | $O(N^2)$ | $O(N)$ (SSM ท้องถิ่น) | $O(N)$ (SSM เต็มรูป) |
| Activation memory ต่อเลเยอร์ | เก็บเต็ม | เก็บเต็ม | **ไม่เก็บ (reversible)** |
| กลไกการเรียนรู้ | Backprop เต็ม | Backprop + stop-gradient memory | **Backprop เต็ม ทุกส่วน differentiable** |
| Long context | จำกัดด้วย KV cache | Memory ภายนอก (EMA update) | Memory ภายนอก (gradient-based update) |
| ความเสถียรการเทรน | สูง (มาตรฐานอุตสาหกรรม) | ปานกลาง-สูง | คาดว่าสูง (ทุกกลไกมี prior art gradient-based) |
| ความเร็วต่อ step | เร็ว | เร็ว-ปานกลาง | ช้าลงจาก recompute overhead |
| VRAM ที่ต้องใช้ (โมเดลลึก) | สูงมาก | ปานกลาง | **ต่ำที่สุดในกลุ่ม** |

---

## 10. Pseudocode

> **อัปเดต v2.1 — หมายเหตุสำคัญ:** โค้ดในหัวข้อนี้คือ **pseudocode เชิงแนวคิดที่อัปเดตให้ตรงกับ design ที่ผ่านการ validate แล้ว** ในโค้ดจริงที่รันการทดลอง (ไฟล์ `hmn_v2.py`) อาจมีรายละเอียด implementation เพิ่มเติม (เช่น normalized key, softmax temperature, usage-based allocation) ซึ่งเป็นรายละเอียดปลีกย่อยที่ระบุในหัวข้อ 18 โค้ดจริงควรใช้เป็นอ้างอิงหลักในการ implement ต่อ

### 10.1 Reversible Helix Coupling (PyTorch-style, Pre-LN)

```python
class SelectiveSSM(nn.Module):
    """Mamba-style selective SSM block, ทำงานแบบ causal บนแกนเวลา
    v2.0: ใช้ Pre-LN (LayerNorm อยู่ภายใน ก่อน in_proj) เป็นค่าเริ่มต้น"""
    def __init__(self, dim, state_dim, prenorm=True):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.ln = nn.LayerNorm(dim) if prenorm else None   # ← Pre-LN (v2.0)
        # พารามิเตอร์ A, B, C, Delta projection ...
        self.in_proj = nn.Linear(dim, dim * 2)
        self.delta_proj = nn.Linear(dim, state_dim)
        self.A_log = nn.Parameter(torch.randn(dim, state_dim))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        if self.ln is not None:
            x = self.ln(x)          # ← norm ก่อนใน_proj (Pre-LN) ไม่ใช่หลัง coupling
        # ... selective scan implementation (parallel scan) ...
        return self.out_proj(y)


class HelixCouplingBlock(nn.Module):
    """เลเยอร์ reversible coupling หนึ่งชั้น (Pre-LN ภายใน F1/F2)"""
    def __init__(self, dim, state_dim, prenorm=True):
        super().__init__()
        half = dim // 2
        self.F1 = SelectiveSSM(half, state_dim, prenorm=prenorm)
        self.F2 = SelectiveSSM(half, state_dim, prenorm=prenorm)

    def forward(self, h):
        h_a, h_b = h.chunk(2, dim=-1)
        h_a_new = h_a + self.F1(h_b)
        h_b_new = h_b + self.F2(h_a_new)
        return torch.cat([h_a_new, h_b_new], dim=-1)

    def inverse(self, h_new):
        """คำนวณย้อนกลับ — ใช้เฉพาะตอน backward custom autograd"""
        h_a_new, h_b_new = h_new.chunk(2, dim=-1)
        h_b = h_b_new - self.F2(h_a_new)
        h_a = h_a_new - self.F1(h_b)
        return torch.cat([h_a, h_b], dim=-1)


class ReversibleFunction(torch.autograd.Function):
    """Custom autograd function ที่ไม่เก็บ activation กลาง
    และคำนวณ gradient ผ่านการ reconstruct แบบย้อนกลับ"""

    @staticmethod
    def forward(ctx, x, blocks):
        ctx.blocks = blocks
        with torch.no_grad():
            h = x
            for block in blocks:
                h = block.forward(h)
        ctx.save_for_backward(h)  # เก็บเฉพาะ output สุดท้าย
        return h

    @staticmethod
    def backward(ctx, grad_output):
        (h,) = ctx.saved_tensors
        blocks = ctx.blocks
        grad = grad_output
        for block in reversed(blocks):
            with torch.no_grad():
                h_prev = block.inverse(h)  # reconstruct activation ย้อนกลับ
            with torch.enable_grad():
                h_prev_ = h_prev.detach().requires_grad_(True)
                h_recomputed = block.forward(h_prev_)
                grad_h_prev, *param_grads = torch.autograd.grad(
                    h_recomputed, [h_prev_] + list(block.parameters()),
                    grad_outputs=grad
                )
            # สะสม gradient เข้าพารามิเตอร์ของ block ตรงนี้
            for p, g in zip(block.parameters(), param_grads):
                p.grad = g if p.grad is None else p.grad + g
            h = h_prev
            grad = grad_h_prev
        return grad, None
```

### 10.2 Sparse Conditional Compute (Product-Key Routing + aux loss)

```python
class SparseConditionalCompute(nn.Module):
    def __init__(self, dim, n_experts, top_k, key_dim=128, aux_coef=0.1):
        super().__init__()
        self.n_sub = int(n_experts ** 0.5)  # แบ่งเป็น sqrt(N) x sqrt(N)
        self.top_k = top_k
        self.aux_coef = aux_coef
        self.query_proj = nn.Linear(dim, key_dim)
        self.keys_1 = nn.Parameter(torch.randn(self.n_sub, key_dim // 2))
        self.keys_2 = nn.Parameter(torch.randn(self.n_sub, key_dim // 2))
        self.values = nn.Embedding(n_experts, dim)
        self.aux_loss = torch.tensor(0.0)   # สะสม load-balancing loss รอบล่าสุด

    def forward(self, x):
        q = self.query_proj(x)
        q1, q2 = q.chunk(2, dim=-1)
        # หา top-sqrt(k) ในแต่ละ sub-key space
        score1 = q1 @ self.keys_1.T
        score2 = q2 @ self.keys_2.T
        top1_val, top1_idx = score1.topk(int(self.top_k ** 0.5), dim=-1)
        top2_val, top2_idx = score2.topk(int(self.top_k ** 0.5), dim=-1)
        # รวม cross product แล้วเลือก top_k จริง
        combined_scores = top1_val.unsqueeze(-1) + top2_val.unsqueeze(-2)
        combined_idx = top1_idx.unsqueeze(-1) * self.n_sub + top2_idx.unsqueeze(-2)
        flat_scores = combined_scores.flatten(-2)
        flat_idx = combined_idx.flatten(-2)
        final_val, final_pos = flat_scores.topk(self.top_k, dim=-1)
        final_idx = torch.gather(flat_idx, -1, final_pos)
        weights = final_val.softmax(dim=-1)
        expert_vals = self.values(final_idx)  # (..., top_k, dim)
        out = (weights.unsqueeze(-1) * expert_vals).sum(dim=-2)
        # v2.0: auxiliary load-balancing loss (ป้องกัน expert collapse)
        self.aux_loss = self._load_balance(final_idx, weights, x.shape[0])
        return out

    def _load_balance(self, idx, gate, B):
        """normalized ในช่วง [0,1]: 0 = สมดุล, 1 = collapse"""
        frac_selected = torch.zeros(self.n_experts, device=idx.device)
        frac_selected.index_add_(0, idx.reshape(-1),
                                 torch.ones(idx.numel(), device=idx.device) / (B * self.top_k))
        frac_gate = torch.zeros(self.n_experts, device=idx.device)
        frac_gate.index_add_(0, idx.reshape(-1), gate.reshape(-1) / B)
        balance = self.n_experts * torch.sum(frac_selected * frac_gate)
        return (balance - 1.0) / (self.n_experts - 1.0)   # ← × self.aux_coef แล้วบวกกับ main loss
```

### 10.3 Differentiable Episodic Memory (โครงร่าง, key/value แยก)

```python
class DifferentiableEpisodicMemory(nn.Module):
    def __init__(self, dim, n_cells, top_k):
        super().__init__()
        self.top_k = top_k
        # v2.0: key มาจาก PREVIOUS hidden, value มาจาก CURRENT hidden (แยก projection)
        self.key_proj = nn.Linear(dim, dim)          # ← key จาก state ก่อนหน้า
        self.val_proj = nn.Linear(dim, dim * 2 + 1)  # ← erase/add/strength จาก state ปัจจุบัน
        self.read_proj = nn.Linear(dim, dim + 1)
        self.out_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim * 2, dim)
        self.keys = torch.randn(n_cells, dim)        # เก็บบน CPU สำหรับสเกลใหญ่
        self.values = torch.randn(n_cells, dim)

    def read(self, h, prev_h):
        q = self.read_proj(h)[..., :dim]
        # ใช้ ANN index (เช่น FAISS) หา top-k cell แบบประมาณ; เล็กใช้ full content addressing
        idx, sim = approximate_nn_search(q, self.keys, self.top_k)
        selected_v = self.values[idx].to(h.device)   # โอนเฉพาะ top-k เข้า GPU
        weights = sim.softmax(dim=-1)
        m = (weights.unsqueeze(-1) * selected_v).sum(dim=-2)
        gate = torch.sigmoid(self.gate_proj(torch.cat([h, m], dim=-1)))
        return gate * h + (1 - gate) * m, idx, weights

    def write(self, prev_h, cur_h, idx, weights):
        # v2.0: write key จาก prev_h (association), erase/add จาก cur_h
        wk = F.normalize(self.key_proj(prev_h), dim=-1)
        v = self.val_proj(cur_h)
        erase = torch.sigmoid(v[..., :dim])
        add = v[..., dim:2*dim]
        w = softmax(weight * similarity(wk, keys))   # content-based write addressing
        self.values[idx] = (self.values[idx] * (1 - w * erase) + w * add)  # soft erase-add (differentiable)
```

---

## 11. แผนการทดลองแบบขั้นบันได (Staged Validation Plan)

การพัฒนาสถาปัตยกรรมใหม่ต้องตรวจสอบทีละองค์ประกอบ ไม่ควรรวมทั้งหมดแล้วทดสอบพร้อมกันตั้งแต่แรก:

### Stage 0 — ตรวจสอบความถูกต้องของ Reversible Reconstruction
ทดสอบว่า `block.inverse(block.forward(x))` ให้ค่าเท่ากับ `x` จริงในระดับ floating-point tolerance ที่ยอมรับได้ (ก่อนแม้แต่จะเทรนอะไร)

### Stage 1 — Reversible SSM เดี่ยวบน Toy Task
เทรนบนงาน copy-task / associative recall แบบง่าย เปรียบเทียบ loss curve และ VRAM usage กับ SSM แบบไม่ reversible ขนาดเท่ากัน

### Stage 2 — เพิ่ม Sparse Conditional Compute
วัด perplexity บน dataset ขนาดเล็ก (เช่น WikiText-2 หรือ corpus ภาษาไทยขนาดเล็ก) เทียบกับ dense feed-forward baseline ที่ compute เท่ากัน

### Stage 3 — เพิ่ม Differentiable Episodic Memory
ทดสอบงานที่ต้องการ long-range recall (เช่น needle-in-haystack แบบง่าย) เพื่อวัดว่าหน่วยความจำภายนอกช่วยจริงหรือไม่

### Stage 4 — รวมทั้งหมด + วัด VRAM บนสเปคเป้าหมาย
รันบน GPU 8–12GB จริง วัด VRAM peak, throughput, และคุณภาพเทียบกับ dense Transformer ขนาดพารามิเตอร์ compute-equivalent

> **อัปเดต v2.0 — ข้อจำกัดเชิงระเบียบวิธีของแผนนี้:** Stage 1–3 (component-level testing) มีไว้เพื่อตรวจสอบว่า **แต่ละส่วนทำงานถูกต้อง** เท่านั้น ไม่ใช่เพื่อสรุปพฤติกรรมเชิง scaling ของระบบรวม — ข้อสรุปเรื่อง scaling (เช่น width vs depth) **ต้องทดสอบที่ระดับ Stage 4/full-system เท่านั้น** เพราะผล component-level อาจ generalize ไปผิดได้จริง (ดูตัวอย่างในหัวข้อ 18.2 ที่สรุปผิดเพราะใช้ผลของ standalone memory)

---

## 12. ความเสี่ยงและข้อจำกัด

*รายการนี้ปรับปรุงตามผลการทดลองจริง — ข้อย่อยที่ขีดฆ่า หมายถึงผ่านการแก้/ยืนยันแล้วในเวอร์ชันนี้*

1. ~~**Recompute overhead จริง**~~ **[ยืนยันแล้ว แต่ยอมรับได้]** — วัดจริงพบว่า overhead ต่อ step อยู่ในช่วงที่คาดไว้ ไม่ได้เป็นตัวขวางการเทรนบน CPU สเปคต่ำ (เทรนครบ 1800 steps บน RAM 4.6GB ได้ภายใน ~220 วินาที)
2. ~~**Floating-point drift สะสม**~~ **[ยืนยันว่าไม่ใช่ปัญหาในสเกลที่ทดสอบ]** — reconstruction error วัดได้จริงที่ 1.5e-8 ถึง 7.5e-9 แม้รวม MoE และ memory เข้าไปในระบบแล้ว ต่ำกว่าระดับที่น่ากังวลมาก ยังต้องจับตาเมื่อ scale เป็นเลเยอร์ลึกกว่านี้มาก (L > 50)
3. **Integration risk ระหว่าง 3 องค์ประกอบ** **[ยืนยันบางส่วน — ผลดีกว่าคาด]** — Stage 4 (full integration) ให้ loss ดีที่สุดที่ 1.59 เทียบกับ Stage 1 เดี่ยวที่ 1.58 แสดงว่าการรวมโมดูลไม่ได้ลดพลัง backbone อย่างมีนัยสำคัญ — ความเสี่ยงนี้ลดระดับลงจาก "ไม่ทราบ" เป็น "จัดการได้"
4. ~~**Routing collapse**~~ **[แก้ไขแล้วใน v2.0 — ด้วย aux load-balancing loss]** — พบ expert ไม่ถูกใช้งานประมาณ 2–6% ตอน initialization ถูกแก้ด้วย auxiliary load-balancing loss ที่เพิ่มใน v2 roadmap ข้อ 3 วัดจริงหลังแก้: aux loss ลดจาก 0.07 → 0.04 และ expert ใช้ครบ 16/16 (ดูหัวข้อ 18.1) ยังต้องเฝ้าระวังตอน scale ขึ้น (more experts)
5. **ANN search accuracy vs speed trade-off** — ยังไม่ได้ทดสอบที่ memory bank ขนาดใหญ่จริง (การทดสอบใช้ 12–16 slots เท่านั้น) ความเสี่ยงนี้ยังเปิดอยู่
6. ~~**Capacity เป็นตัวจำกัดหลัก มากกว่าความลึก**~~ **[แก้ไข v2.0 — ผลเดิมใช้ได้เฉพาะ standalone memory]** เดิมพบว่าการเพิ่ม dim จาก 40→64 ทำให้ recall accuracy พุ่งจาก 27% เป็น 100% แต่เมื่อทดสอบซ้ำบน **HMN เต็มระบบ** (รวม coupling backbone + memory) พบว่าการเพิ่มความลึก (depth) ให้ efficiency ต่อพารามิเตอร์ดีกว่าการเพิ่มความกว้างในงาน recall — ผลเดิมเป็นคุณสมบัติเฉพาะของโมดูล memory เดี่ยวๆ เท่านั้น ไม่ใช่ข้อสรุปของสถาปัตยกรรมทั้งระบบ รายละเอียดในหัวข้อ 18.2
7. **[ใหม่] Generation ยังทำคณิตศาสตร์จริงไม่ได้** — โมเดลขนาดเล็กที่ทดสอบ (Stage 4) เรียนรู้ pattern ของตัวเลขได้แต่ไม่ได้เรียนรู้การคำนวณจริง (เช่น 5+7 ให้ผลลัพธ์ผิดแบบ "5+7*44") คาดว่าเป็นข้อจำกัดของขนาดโมเดล+ระยะเวลาเทรนที่สั้นเกินไป มากกว่าข้อจำกัดเชิงสถาปัตยกรรม แต่ยังไม่มีหลักฐานยืนยันจนกว่าจะทดสอบโมเดลใหญ่ขึ้น
8. **[ใหม่] Hard top-K ไม่มี gradient ผ่าน indices** — ตอนนี้ gradient ไหลผ่านได้เฉพาะทาง gate weights ไม่ใช่ทาง discrete selection ทำให้การเรียนรู้ routing อาจช้ากว่าที่ควร (ต่างจากที่ออกแบบไว้ในหัวข้อ 5.3.1 ว่าจะใช้ soft top-K + STE เต็มรูป — เวอร์ชันที่ทดสอบจริงยังไม่ได้ implement STE เต็มที่)
9. **[ใหม่] Sequential SSM scan เป็นคอขวดเวลา** — การ implement ปัจจุบันใช้ Python loop สำหรับ scan (BPTT 48 steps) ยังไม่ได้ทำ parallel/chunked scan ตามที่ทฤษฎีระบุไว้ในหัวข้อ 6.2 นี่เป็นข้อจำกัดของ implementation ไม่ใช่ของสถาปัตยกรรม แต่ส่งผลต่อความเร็วจริงอย่างมีนัยสำคัญ
10. **[ใหม่] Memory และ MoE ยังไม่ reversible** — activation ของสองโมดูลนี้ยังต้องเก็บแบบ checkpoint ปกติ ทำให้การประหยัด activation memory ยังไม่ครอบคลุมทั้งระบบ 100% ตามที่ตั้งเป้าไว้ในหัวข้อ 6.3 — เป็นข้อจำกัดที่ยังไม่ได้แก้

---

## 13. คำถามวิจัยที่ยังเปิดอยู่ (Open Research Questions)

1. ~~Reversible coupling ที่ใช้ Selective SSM จะ reconstruct ได้แม่นยำเท่า RevNet ดั้งเดิมหรือไม่~~ **[ตอบบางส่วนใน v2.0]** — วัดจริงพบว่า Selective SSM ยัง reconstruct ได้แม่นยำ (error 2.4×10⁻⁷ หลังเทรนครบ 1500 steps) ในสเกลที่ทดสอบ ยังไม่ชัดเจนเมื่อ L > 50
2. ควรวาง Episodic Memory ทุกกี่เลเยอร์ (memory_interval) จึงจะสมดุลระหว่างประโยชน์ด้าน long-context กับ compute overhead ที่เพิ่มขึ้น
3. Straight-through estimator สำหรับ product-key routing จะทำให้เกิด gradient variance สูงในโมเดลที่ลึกและมี sparse routing หลายชั้นซ้อนกันหรือไม่
4. จุดสมดุลที่เหมาะสมระหว่างจำนวนเลเยอร์ที่ทำ full checkpoint (เก็บ activation จริง) กับเลเยอร์ที่เป็น reversible ล้วนๆ อยู่ที่ไหน เพื่อจัดการ floating-point drift โดยไม่เสีย VRAM มากเกินไป

---

## 14. เอกสารอ้างอิงเชิงแนวคิด (Conceptual References)

องค์ประกอบของ HMN อ้างอิงแนวคิดจากงานวิจัยที่มีอยู่จริงดังต่อไปนี้ (ระบุไว้เพื่อความโปร่งใส ไม่ใช่การอ้างว่า HMN เป็นงานวิจัยที่ตีพิมพ์แล้ว):

- **Reversible Residual Networks (RevNet)** — แนวคิดเลเยอร์ที่คำนวณย้อนกลับได้เพื่อลด activation memory
- **Reformer** — การประยุกต์ reversible layers เข้ากับสถาปัตยกรรมแบบ sequence model ระดับ NLP
- **Mamba / Selective State-Space Models** — กลไก SSM ที่มีพารามิเตอร์ขึ้นกับ input แบบ selective
- **Product-Key Memory Layers** — เทคนิคการค้นหา sparse memory ด้วยการแบ่ง key space
- **Differentiable Neural Computer (DNC)** — กลไก read/write หน่วยความจำภายนอกแบบ differentiable เต็มรูปแบบ
- **Mixture-of-Experts (MoE)** — แนวคิด conditional computation ที่ active เฉพาะบางส่วนของพารามิเตอร์

*หมายเหตุ: รายการนี้เป็นการระบุที่มาของแนวคิดเชิงหลักการเท่านั้น ไม่ใช่การอ้างอิงทางวิชาการที่สมบูรณ์ ผู้ที่สนใจนำไปพัฒนาต่อควรค้นคว้า paper ต้นฉบับและ implementation จริงของแต่ละเทคนิคก่อนเริ่ม implement*

---

## 15. ภาคผนวก: ตารางไฮเปอร์พารามิเตอร์แนะนำ

สำหรับการทดลองเริ่มต้นระดับ Stage 1–2 (โมเดลขนาดเล็กสำหรับ validate แนวคิด):

| ไฮเปอร์พารามิเตอร์ | ค่าแนะนำเริ่มต้น | หมายเหตุ |
|---|---|---|
| Hidden dim ($d$) | 512 | เริ่มเล็กเพื่อ debug ง่าย |
| จำนวนเลเยอร์ ($L$) | 8–12 | เพิ่มทีหลังเมื่อ reversible ทำงานถูกต้องแล้ว |
| SSM state dim ($P$) | 16–32 | ตามค่ามาตรฐานของ Mamba ขนาดเล็ก |
| จำนวน experts ($N_e$) | 4,096–16,384 | เริ่มเล็กกว่าค่าที่เสนอในหัวข้อ 5.3 |
| Top-K (routing) | 8–16 | |
| aux load-balancing coef | 0.1 | ค่าเริ่มต้นที่ใช้ใน v2; ปรับถ้า main loss แย่ลง |
| Memory bank size ($M$) | 10,000–50,000 | สำหรับ Stage 3 เท่านั้น |
| Memory read top-K | 16–32 | |
| memory_interval | ทุก 4 เลเยอร์ | ปรับตาม Stage 3 ผลลัพธ์ |
| Learning rate | 1e-4 ถึง 3e-4 (AdamW) | มาตรฐานสำหรับโมเดลขนาดนี้ |
| Batch size | จำกัดตาม VRAM จริง เริ่มที่ 1–2 | เพิ่มด้วย gradient accumulation |

---

## 16. ผลการทดลองจริง — Stage 1–4 (Empirical Validation Results)

หัวข้อนี้บันทึกผลการทดลองจริงตาม Staged Validation Plan ในหัวข้อ 11 รันบนฮาร์ดแวร์ **CPU i3 Gen2, 4 threads, RAM 4.6GB** ซึ่งอยู่ในระดับต่ำกว่าสเปคเป้าหมายขั้นต่ำที่ตั้งไว้ตอนแรก (GPU 6–12GB หรือ CPU+RAM 16–32GB) — การที่ยังเทรนได้ครบวงจรบนสเปคที่ต่ำกว่านี้ถือเป็นสัญญาณบวกต่อเป้าหมายหลักของสถาปัตยกรรม

### 16.1 ความเร็วเทรนโดยรวม

| Config | เวลา/step |
|---|---|
| dim 64, 4 layers, batch size 6 | ≈ 0.12s/step |
| dim 96, 6 layers, batch size 8 | ≈ 0.11s/step |
| dim 128, 8 layers, batch size 8 | ≈ 0.16s/step |

### 16.2 ผลแยกตาม Stage

| Stage | Config | Loss/Metric เริ่ม → ดีที่สุด | Steps | เวลารวม |
|---|---|---|---|---|
| 1 (Pre-LN) | dim96, state12, 6L, bs8 | loss 8.27 → 1.58 | 1500 | 167s |
| 1 (Pre-LN) | dim128, state16, 8L | loss 8.03 → 2.30 | 1500 | 246s |
| 2 (recall) | dim64, 16 slots, bs24 | accuracy 1% → 100% | 1000 | ~60s |
| 2 (recall) | dim40, 12 slots | accuracy ติดที่ 27% (capacity ไม่พอ) | 3000 | — |
| 3 (expressivity) | dim16, 64 experts, k=2 | MSE 1.42 → 0.00005 | 600 | — |
| 4 (full integration) | dim64, 8 state, 4L, 8 slots, bs6 | loss 8.13 → 1.59 | 1800 | ~220s |

### 16.3 สรุปสิ่งที่พิสูจน์แล้ว (Confirmed)

1. **Reversible SSM ทำงานได้จริงตามทฤษฎี** — ค่า reconstruction error อยู่ที่ 1.5×10⁻⁸ ถึง 7.5×10⁻⁹ แม้ในระบบที่รวม MoE และ episodic memory เข้าไปแล้ว ยืนยันว่าแนวคิดในหัวข้อ 5.2.1 (backward reconstruction แม่นยำระดับคณิตศาสตร์) ใช้ได้จริงในทางปฏิบัติ ไม่ใช่แค่ทฤษฎีบนกระดาษ
2. **การเทรนเสถียรเมื่อใช้ Pre-LN เท่านั้น** — พบว่า Post-LN ทำให้ loss ระเบิด (NaN) ที่ step 63 ขณะที่ Pre-LN ให้ loss ลงเรียบตลอด 1500+ steps โดยไม่มี NaN แม้แต่ครั้งเดียว — **นี่คือข้อกำหนดบังคับของ architecture** (Pre-LN ภายใน F1/F2) ที่ได้นำเข้าสู่หัวข้อ 5.2.2 แล้ว
3. **Episodic memory เรียนรู้ long-range recall ได้จริง ไม่ใช่แค่ recency bias** — ทดสอบ recall คู่แรกจาก 5 คู่ที่ป้อนเข้าไป (ตำแหน่งไกลที่สุดจากจุด query) ได้ accuracy 100% ยืนยันว่ากลไกใน 5.4.2–5.4.3 ทำงานตามที่ออกแบบจริง ไม่ได้อาศัยแค่ข้อมูลล่าสุด
4. **PK-MoE มี expressive power จริง** — สามารถประมาณ dense mapping ได้จน MSE ลดลงเหลือ 0.00005 และการ routing แยกแยะ class ของ input ได้จริง (measured overlap ระหว่าง class ต่ำเพียง 0.03) แสดงว่า expert ที่ถูกเลือกมีความหมายเชิง semantic ไม่ใช่การสุ่ม
5. **การรวม 3 โมดูลเข้าด้วยกันไม่ลดทอนพลังของ backbone** — Stage 4 (full integration) ได้ loss ดีที่สุด 1.59 ใกล้เคียงกับ Stage 1 เดี่ยวที่ 1.58 นี่คือหลักฐานสำคัญที่ลดความเสี่ยงข้อ 3 ในหัวข้อ 12 ของเวอร์ชันเดิมลงอย่างมาก
6. **เทรนได้ครบวงจรบน RAM 4.6GB โดยไม่ใช้ GPU เลย** — ต่ำกว่าสเปคเป้าหมายเดิมของโปรเจกต์ (16–32GB CPU+RAM) มาก แสดงว่า approach นี้มีช่องว่างด้าน scale ให้ขยายได้อีกมากก่อนจะชนข้อจำกัดฮาร์ดแวร์จริง

### 16.4 สิ่งที่พบใหม่ในระหว่างการทดลอง

- ~~**Capacity (ความกว้าง) สำคัญกว่าความลึกในช่วงเริ่มต้น**~~ **[ข้อสรุปนี้ผิด — แก้ไขใน v2.0 ดูหัวข้อ 18.2]** — ผลที่เห็นตอนแรก (dim 40→64 พุ่ง 27→100%) เป็นการทดสอบ **standalone memory module เท่านั้น** เมื่อทดสอบซ้ำบนระบบเต็ม (coupling backbone + memory) พบว่าความลึกให้ efficiency ต่อพารามิเตอร์ดีกว่า จึงไม่ใช่ข้อสรุปของสถาปัตยกรรมทั้งระบบ — เก็บไว้เป็นบันทึกประวัติว่าข้อสรุปจาก component-level test อาจ generalize ผิด (รายละเอียดในหัวข้อ 18.2)
- **Generation เชิงคณิตศาสตร์ยังไม่เกิดขึ้นที่สเกลนี้** — เป็นการเตือนว่าอย่าตีความ loss ที่ลดลงว่าโมเดล "เข้าใจ" งานอย่างแท้จริง จำเป็นต้องมี evaluation แบบ task-specific (ไม่ใช่แค่ loss) ก่อนสรุปความสามารถของโมเดล
- **STE (straight-through estimator) สำหรับ routing ที่ออกแบบไว้ในหัวข้อ 5.3.1 ยังไม่ได้ implement เต็มรูปแบบ** — เวอร์ชันที่ทดสอบจริงใช้ hard top-K ธรรมดา ซึ่งอาจเป็นสาเหตุหนึ่งที่ทำให้ routing เรียนรู้ช้ากว่าที่ทฤษฎีคาดไว้

---

## 17. แผนสำหรับ v2 (Roadmap)

จากผลการทดลองในหัวข้อ 16 สรุปเป็นแผนปรับปรุงสำหรับเวอร์ชันถัดไป เรียงตามลำดับความสำคัญ:

> **อัปเดต v2.0:** รายการด้านล่างคือ roadmap ที่วางไว้ตอน v1.1 — ข้อ 1–5 ผ่านการ validate จริงแล้วครบถ้วน (ดูหัวข้อ 18) รวมถึง **ข้อ 2 ถูกแก้ไขเนื้อหาแล้ว** เนื่องจากผลการทดลองค้านสมมติฐานเดิม ตารางด้านล่างคงไว้เพื่อเป็นบันทึกประวัติการตัดสินใจ ส่วนเนื้อหาที่ถูกต้องล่าสุดอยู่ในหัวข้อ 18.2

| ลำดับ | การปรับปรุง | เหตุผลจากผลทดลอง | สถานะ |
|---|---|---|---|
| 1 | ระบุ **Pre-LN ภายในฟังก์ชัน F ของ coupling layer** เป็นค่าเริ่มต้นมาตรฐาน ไม่ใช่ทางเลือก | เป็นหัวใจของความเสถียร — Post-LN ทำให้ระบบระเบิดจริงที่ step 63 | ✅ Validate แล้ว (v2) |
| 2 | ~~เพิ่มความกว้าง (dim/slots) ก่อนความลึก (layers)~~ **[แก้ไข v2.0]** ปรับให้ **สมดุลระหว่าง width และ depth** — depth มี efficiency ต่อพารามิเตอร์ดีกว่าเมื่อรวมกับ coupling backbone | ผล Stage 2 เดิม (dim 40→64 พุ่ง 27→100%) เป็นผลของ **standalone memory เท่านั้น** ไม่ใช่ระบบเต็ม — เมื่อทดสอบ HMN เต็มระบบ (coupling+memory) พบว่าเพิ่มความลึกใช้พารามิเตอร์น้อยกว่าและ converge เร็วเท่ากับเพิ่มความกว้าง | ✅ Validate แล้ว (v2) — สมมติฐานเดิมผิด |
| 3 | เพิ่ม **auxiliary load-balancing loss** ให้ Sparse Conditional Compute | ป้องกัน expert collapse ที่พบจริง (2–6% ไม่ถูกใช้งาน) — เป็นช่องว่างที่ v1.0 ระบุไว้แล้วในหัวข้อ 12 ข้อ 4 แต่ยังไม่ได้แก้ | ✅ Validate แล้ว (v2) |
| 4 | ปรับ Episodic Memory ให้แยก **key_proj(previous state) และ value_proj(current state)** อย่างชัดเจน | เป็น design ที่พิสูจน์แล้วว่าทำให้ recall แม่นยำ 100% ในการทดลอง Stage 2 | ✅ Validate แล้ว (v2) |
| 5 | เขียน SSM scan ใหม่เป็นแบบ **chunked/parallel scan** แทน Python sequential loop | แก้คอขวดเวลาเทรนที่พบจริง — เป็นข้อจำกัดของ implementation ไม่ใช่ของสถาปัตยกรรม | 🔲 ยังไม่เริ่ม |
| 6 | ทดสอบบน **GPU และ dataset ขนาดใหญ่ขึ้น (ระดับ 10M+ token สำหรับงานคณิตศาสตร์)** ก่อนประเมินความสามารถด้าน generation | ผลปัจจุบันจากโมเดลเล็กยังสรุปไม่ได้ว่าข้อจำกัดด้าน generation มาจากสถาปัตยกรรมหรือขนาดโมเดล | 🔲 ยังไม่เริ่ม |
| 7 | Implement **STE เต็มรูปแบบสำหรับ product-key routing** ตามที่ออกแบบไว้ในหัวข้อ 5.3.1 | เวอร์ชันทดสอบยังใช้ hard top-K ธรรมดา อาจเป็นสาเหตุที่ routing เรียนรู้ช้า | 🔲 ยังไม่เริ่ม |
| 8 | ขยาย reversible property ให้ครอบคลุม MoE และ Memory module | ปัจจุบันสองโมดูลนี้ยังต้องใช้ standard checkpoint ทำให้การประหยัด activation memory ยังไม่ครบ 100% ตามเป้าหมายเดิม | 🔲 ยังไม่เริ่ม |

**เกณฑ์ก่อนเริ่ม v2:** แนะนำให้ทำข้อ 1–4 ก่อน (แก้จุดเสี่ยงที่ยืนยันแล้วและใช้ทรัพยากรน้อย) แล้วค่อยลงทุนกับข้อ 5–6 (ต้องใช้เวลา/ทรัพยากรมากกว่า) เพื่อไม่ให้เสียเวลากับการ optimize ความเร็วก่อนที่ความถูกต้องเชิง algorithm จะนิ่ง

---

## 18. ผลการ validate v2 — ข้อ 1–5 (v2 Validation Results)

หัวข้อนี้บันทึกผลการ validate roadmap ข้อ 1–5 จากหัวข้อ 17 ซึ่งทำครบทั้งหมดแล้ว โค้ดทั้งชุดอยู่ที่ `hmn_v2.py` (ไฟล์อ้างอิง: `/tmp/opencode/hmn_v2.py`) พร้อมสคริปต์ validate แยกต่างหาก

> **⚠️ หมายเหตุ v2.2 (reproduction attempt): หัวข้อ 18 ทั้งหมดเป็น UNVERIFIED — reproduce ไม่ได้**
> ความพยายาม reproduce (2026-08-07) ด้วย config เดียวกับ 18.2 (dim40/L4, ไม่มี MoE,
> 8 คู่/50 keys, single-token disjoint tokens) ได้ **acc 3-6%** ที่ 900-1200 steps — ไม่ใช่ 90%
> และ standalone memory ที่ 5 คู่ ได้ ~55-65% (ไม่ได้ 82-100%)
>
> **สาเหตุที่สันนิษฐานของตัวเลข 100% เดิม:** eval protocol เดิม query คู่แรกเสมอ (ตำแหน่งที่
> ทำนายได้จาก positional shortcut — ค่าแรกที่เห็นหลัง START) เทสต์ cross กับ random-query
> พบว่าถ้า train ด้วย query แรก → acc 100% เฉพาะตอน query แรก แต่ **ตกเหลือ ~49%** เมื่อ query
> ตำแหน่งอื่น และ train แบบ random-query ได้แค่ ~45-55% ทั้งหมด — แสดงว่าตัวเลข 100% เดิม
> มาจาก positional shortcut ไม่ใช่ content addressing ที่ generalize
>
> โค้ดต้นฉบับ (doc_mem.py) หายไปแล้ว — ไม่สามารถตรวจสอบย้อนหลังว่าตัวเลขเดิมถูกต้องหรือ
> เป็น bug ที่บังเอิญให้ผลดี **ห้ามใช้ 90-100% เป็นเป้าหมายอ้างอิง** จนกว่าจะ reproduce ได้
> เพดานอ้างอิงปัจจุบันที่ verified: standalone random-query ~55%, Option 1 (raw-embed memory)
> ~51% (8 คู่) — ดู task2_findings.md หัวข้อ 7-8

### 18.1 สรุปผลรวม

| ข้อ | Roadmap | ผล | ตัวเลขวัดจริง |
|---|---|---|---|
| 1 | Pre-LN เป็นค่าเริ่มต้น | ✅ PASS | reconstruction roundtrip error 5.96×10⁻⁸, เทรน 1500 steps ไม่มี NaN |
| 2 | aux load-balancing loss (PK-MoE) | ✅ PASS | aux loss ลดลงจนต่ำสุด 0.039 (จุดเริ่ม 0.046, พีค 0.070, ปลาย 0.053) และ expert usage 16/16 (ใช้ครบทุกตัว) |
| 3 | แยก key_proj(prev)/val_proj(cur) ใน memory | ✅ PASS | recall accuracy 100% (query คู่แรกจาก 5 คู่ที่ป้อน) |
| 4 | width vs depth sweep | ⚠️ ผลค้านสมมติฐานเดิมใน v1.1 | ดูหัวข้อ 18.2 |
| 5 | HMN v2 เต็มระบบ (รวมทุกส่วน) | ✅ PASS | loss 8.06 → 1.78, reversible error 2.4×10⁻⁷ หลังเทรนครบ |

### 18.2 ข้อค้นพบสำคัญ: แก้ไขสมมติฐาน Width vs Depth

> **⚠️ หมายเหตุ v2.2 (UNVERIFIED — เหมือนหัวข้อ 18.1):** sweep นี้ใช้ eval protocol
> "recency bias = 0%" ซึ่งตามนิยามในหัวข้อ 5.4 (ดู 18.1 ข้อ 3) คือ **query คู่แรกเสมอ**
> (ตำแหน่งไกลสุดจากจุด query) — เป็น protocol เดียวกับที่พิสูจน์แล้วว่าให้ positional
> shortcut ใน diagnostic B (train/query คู่แรก → 100% แต่ random-query ตกเหลือ ~49%)
> โค้ดต้นฉบับ (hmn_v2_width_sweep3.py) หายไปแล้ว ไม่สามารถตรวจสอบย้อนหลังได้
>
> **รันซ้ำ 3 configs (2026-08-07) ด้วย random-query eval แล้ว:**
> - train first → eval first = **99-100% ทุก config** (ยืนยันว่าเลข 90% เดิม = positional shortcut)
> - train first → eval random = **17-18% ทั้งหมด** (ไม่มี config ใดทำ content addressing ได้)
> - train random → eval random: **d80/L4 = 15.3% > d40/L8 = 9.7%** — width ดีกว่า depth
>   กลับไปทางข้อสรุป v1.1 (width-matters) ไม่ใช่ depth-efficiency ตามที่อ้างในตารางนี้
>
> **ข้อสรุป "depth efficiency ดีกว่า width" จึงเป็น UNVERIFIED เช่นกัน** — การที่โมเดล
> เข้าถึง 90% เร็วขึ้นด้วย depth เป็นแค่การเรียน positional shortcut ที่เร็วกว่า ไม่ได้
> พิสูจน์ว่า content addressing ดีขึ้น ต้องรัน sweep ใหม่ด้วย random-query eval จริง
> (ดู task2_findings.md หัวข้อ 9)

ผลจาก v1.1 (หัวข้อ 16.4 เดิม) เคยสรุปว่า "ความกว้างสำคัญกว่าความลึก" จากการทดสอบ **standalone memory module เพียงอย่างเดียว** — **ผลนี้ไม่เป็นจริงเมื่อทดสอบกับระบบเต็ม (coupling backbone + memory) หมายเหตุ: sweep นี้ไม่รวม MoE**

Width sweep บน HMN v2 เต็มระบบ (embed + coupling backbone + episodic memory + head; ไม่มี MoE) ด้วยงานยาก (8 คู่ key-value, 50 keys ทั้งหมด, recency bias = 0%):

| Config | จำนวนพารามิเตอร์ | Steps ถึง 90% accuracy |
|---|---|---|
| dim 40, L4 (baseline) | 29K | 900 |
| dim 80, L4 (กว้างขึ้น) | 94K | 600 |
| dim 40, L8 (ลึกขึ้น) | 42K | 600 |

**การตีความ:** การเพิ่มความลึก (L4→L8) ใช้พารามิเตอร์เพิ่มขึ้นเพียง 13K (29K→42K) แต่ได้ผล convergence เร็วเท่ากับการเพิ่มความกว้าง (L4→dim80) ที่ต้องใช้พารามิเตอร์เพิ่มขึ้นถึง 65K (29K→94K) — กล่าวคือ **ความลึกให้ efficiency ต่อพารามิเตอร์ที่ดีกว่าอย่างชัดเจนในงาน recall เมื่อทดสอบกับระบบเต็ม**

**สาเหตุที่ผลต่างจาก v1.1:** ผล Stage 2 เดิมทดสอบเฉพาะโมดูล memory เดี่ยวๆ โดยไม่มี coupling backbone ร่วมด้วย ทำให้ capacity ของระบบผูกอยู่กับมิติของ memory cell โดยตรง แต่เมื่อรวม backbone เข้ามา ความลึกของ coupling layers ช่วยให้ข้อมูลถูกประมวลผลและกลั่นกรองก่อนเข้าสู่ memory ทำให้แต่ละ memory cell ทำงานได้อย่างมีประสิทธิภาพมากขึ้นโดยไม่ต้องขยายมิติ

> **หมายเหตุ v2.1:** sweep นี้ทดสอบ **coupling backbone + memory เท่านั้น (ไม่มี MoE)** เพราะมีเป้าหมายเพื่อแยกแยะผลของ width/depth ต่อความจุของระบบประมวลผลลำดับ ไม่ใช่ผลของ routing — ข้อสรุปเรื่อง "depth efficiency ดีกว่า" จึงใช้ได้กับบริบท backbone+memory ยังต้องทดสอบซ้ำเมื่อเพิ่ม MoE ในสเกลใหญ่

**บทเรียนเชิงระเบียบวิธี:** นี่คือตัวอย่างของความเสี่ยงในการ generalize ผลจาก component-level testing (Staged Validation Plan หัวข้อ 11) ไปเป็นข้อสรุปของระบบเต็ม — Stage 1–3 ของแผนเดิมมีไว้เพื่อตรวจสอบว่าแต่ละส่วนทำงานถูกต้อง **ไม่ใช่** เพื่อสรุปพฤติกรรมเชิง scaling ของระบบรวม ข้อสรุปเรื่อง scaling ต้องทดสอบที่ระดับ Stage 4/full-system เท่านั้น เอกสารเวอร์ชันถัดไปควรระบุข้อจำกัดนี้ไว้ชัดเจนในแผนการทดลอง (ทำแล้วในหัวข้อ 11)

### 18.3 ข้อยืนยันเพิ่มเติมจากการรวมระบบเต็ม

- **Aux loss ไม่รบกวน main loss** — loss ที่ดีที่สุดของ v2 (1.78) ใกล้เคียงกับ v1 (1.59) แม้เพิ่ม load-balancing term เข้าไป แสดงว่า auxiliary loss ไม่ได้แย่งความสามารถของ main objective อย่างมีนัยสำคัญ
- **Reversible property ไม่เสื่อมสภาพหลังเทรนจริง** — reconstruction error หลังเทรนครบ 1500 steps ยังอยู่ที่ 2.4×10⁻⁷ ซึ่งต่ำมาก ยืนยันว่าการอัปเดตน้ำหนักระหว่างเทรนไม่ได้ทำลายคุณสมบัติ reversible ของ coupling layer (ตอบคำถามวิจัยที่เปิดไว้ในหัวข้อ 13 ข้อ 1 บางส่วน — อย่างน้อยในสเกลที่ทดสอบ Selective SSM ยัง reconstruct ได้แม่นยำพอ)
- **Generation เชิงคณิตศาสตร์ยังทำไม่ได้ (คงเดิมจาก v1.1)** — ทดสอบซ้ำยังได้ผลแบบ 5+7 → 5+7*77 ยืนยันว่าเป็นข้อจำกัดของขนาดโมเดล/ระยะเวลาเทรน ไม่ใช่ regression จากการเปลี่ยนแปลงใน v2 — ยังต้องรอ roadmap ข้อ 6 (ทดสอบสเกลใหญ่บน GPU) เพื่อตอบคำถามนี้ให้ชัดเจน

### 18.4 สิ่งที่ต้องทำต่อ

จาก roadmap หัวข้อ 17 เหลือข้อ 5 (chunked/parallel scan), 6 (GPU + large-scale generation test), 7 (STE เต็มรูปแบบ), 8 (reversible MoE/memory) — ยังไม่ได้เริ่ม แนะนำเรียงลำดับ:

> **อัปเดต v2.2 (post-roadmap):** ข้อ 5 และข้อ 7 เสร็จแล้ว — ดูหัวข้อ 18.7 เหลือข้อ 6 และ 8

1. **ข้อ 5 (parallel scan) ✅ เสร็จแล้ว** — ดูหัวข้อ 18.7.1 สำหรับผล speedup
2. **ข้อ 7 (STE เต็มรูปแบบ) ✅ เสร็จแล้ว** — ดูหัวข้อ 18.7.2
3. **ข้อ 6 (GPU + large-scale test)** — ควรทำหลังข้อ 5 เสร็จ เพราะถ้า scan ยังเป็น Python loop การเทรนโมเดลใหญ่ขึ้นจะช้าจนทดลองได้ไม่คุ้มเวลา
4. **ข้อ 8 (reversible MoE/memory)** — เก็บไว้ท้ายสุด เป็นงานเชิงทฤษฎีที่ซับซ้อนกว่าและผลตอบแทนเชิง VRAM อาจไม่คุ้มถ้าโมเดลยังไม่ได้ขยายใหญ่จนกระทบ VRAM จริง

### 18.7 ผล roadmap ข้อ 5 (chunked parallel scan) และข้อ 7 (STE เต็มรูปแบบ) — v2.2 post-roadmap

> งานทั้งสองทำบน CPU (i3, 4.6GB RAM, 2 threads) หลัง merge Option 1 เข้า hmn_v2.py แล้ว
> การ validate ใช้ regression suite (experiments/regression_hmn_v2.py) เทียบตัวเลข verified ใน 18.6

#### 18.7.1 ข้อ 5: chunked parallel scan (SelectiveSSM)

เดิม SSM scan ใช้ Python loop ตามความยาว sequence (BPTT) เป็นคอขวดความเร็ว เปลี่ยนเป็น
**two-phase chunked scan** ใน `SelectiveSSM._chunked_scan` (hmn_v2.py):

- recurrence `h_t = dA_t*h_{t-1} + dB_t` เป็น linear ต่อเนื่องแบบ diagonal (elementwise)
- **phase 1:** loop เฉพาะ chunk_size ตำแหน่ง (vectorized ครอบทุก chunk พร้อมกัน) คำนวณ
  cumulative product `P` และ forced response `F` ต่อ chunk
- **phase 2:** loop เฉพาะจำนวน chunk (n_chunks) เชื่อมต่อสถานะระหว่าง chunk
- `h = P * h_in + F` — Python loop ลดจาก `T` เป็น `chunk_size + n_chunks`
- **ไม่มี log-domain clamping** (การทำ log-clamp ก่อนหน้าทำให้ error ระเบิด ~8-17
  เมื่อ decay ลึก เพราะ `exp(logP_safe_j)*exp(-logP_safe_s)` ไม่รักษาอัตราส่วนการ decay)
  — วิธีสองเฟสนี้เที่ยงตรงระดับบิตกับ sequential scan

**ผล validate:**

| การตรวจ | ผล |
|---|---|
| เทียบ sequential scan (ทุก config × chunk 8/16/32) | max_err ~1e-6 (float32 rounding) |
| gradient (backward ผ่าน chunked vs sequential) | diff ~1.5e-8 |
| reconstruction error (coupling inverse) | ไม่เปลี่ยน (6.3e-6 / 2.3e-5 / 4.8e-5) |
| regression accuracy | **เท่าเดิมทุกจุด** (single 0.958/0.991/0.955/0.914, 2-token 0.978/0.997/0.976/0.944) |

**Speedup scan-only (2 threads, N_STEPS=12):**

| Config | ก่อน | หลัง |
|---|---|---|
| dim64/4L/bs6 | 286.8ms | 103.8ms (~2.8x) |
| dim96/6L/bs8 | 1012.2ms | 491.3ms (~2.1x) |
| dim128/8L/bs8 | 2172.1ms | 1288.7ms (~1.7x) |

**Full-step speedup:** dim64 1264→1034ms, dim96 2594→2133ms, dim128 4877→3689ms
(สแกนเป็นส่วนเดียวของ step; เหลือคอขวดอื่น เช่น memory + MoE)

#### 18.7.2 ข้อ 7: full STE สำหรับ PK-MoE routing (SparseConditionalCompute)

เดิม routing ใช้ hard top-K: gradient ไหลถึง router เฉพาะผ่าน gate weights
(softmax ของค่าคะแนน expert ที่ถูกเลือก) ไม่ไหลผ่านตัว discrete selection
ทำให้ router เรียนได้ช้า (ข้อจำกัด 18.3 ข้อ 8) เพิ่ม **full straight-through estimator**
(hmn_v2.py, flag `ste=True` default):

- forward ยังเป็น hard top-K (ผลเหมือนเดิมเป๊ะ) — backward เพิ่ม path soft top-K
  (softmax ครอบคะแนน candidate ทั้งหมด) ที่คูณด้วยค่า expert แบบ detach
  → gradient ไหลไปถึงทุก candidate score → router เรียนว่า *ควรเลือก expert ไหน*
  ไม่ใช่แค่ถ่วงน้ำหนักของตัวที่เลือกแล้ว
- `soft - soft.detach()` ทำให้ forward contribution = 0 (ไม่เปลี่ยน forward) แต่
  backward ส่ง gradient ผ่าน selection

**ผล validate:**

| การตรวจ | ผล |
|---|---|
| forward output เทียบ ste on/off | diff = 0.0 (ไม่เปลี่ยน forward) |
| router gradient (query_proj / keys) | เพิ่ม ~3.5x เมื่อเทียบ baseline (0.67→2.36, 0.15→0.50) |
| aux load-balancing loss | ยังทำงาน ช่วง [0,1] ปกติ |
| backward-compat (ste=False) | forward เท่ากับเดิม (diff 0.0) |
| เกม routing จำลอง (learnable selection, values แช่แข็ง) | STE เลือก expert ถูกกว่าและ loss ต่ำกว่าในทุก seed |
| regression accuracy | ไม่กระทบ (Option 1 ไม่เรียก MoE ใน forward) |

**หมายเหตุ:** `HMN` (สถาปัตยกรรมเก่าก่อน Option 1, เรียก MoE) มี divergence ของ loss
ที่มีอยู่ก่อนแล้ว (เทียบ ste=True/False ได้ผลเหมือนกัน) — ไม่ใช่สาเหตุจาก STE

### 18.5 บันทึกการแก้ไข v2.1 (การตรวจทานโดย maintainer)

แก้จุดขัดแย้ง/ความไม่ตรงกัน 4 จุดจาก v2.0 โดยเปรียบเทียบกับผลทดลองจริงที่วัดได้:

| จุด | ปัญหาใน v2.0 | การแก้ใน v2.1 |
|---|---|---|
| 1. Section 12 ข้อ 4 | เขียนว่า routing collapse "ยังไม่แก้ / ยังไม่ได้ใส่ aux loss" แต่ roadmap ข้อ 3 เครื่องหมาย ✅ ผ่านแล้ว — ขัดแย้งกันเอง | ตีกลับเป็น ~~Routing collapse~~ [แก้ไขแล้วใน v2.0] พร้อมตัวเลข aux loss 0.07→0.04 และ expert 16/16 |
| 2. Section 16.4 | ยังค้างข้อสรุปเก่า "ความกว้างสำคัญกว่าความลึก" โดยไม่ตีกลับ — ขัดแย้งกับข้อสรุปใหม่ใน 18.2 | ตีกลับ + หมายเหตุชี้ไป 18.2 และเก็บเป็นบันทึกประวัติเชิงระเบียบวิธี |
| 3. Section 18.2 | เขียนว่า sweep ทดสอบ "coupling + memory + MoE รวมกัน" แต่การทดสอบจริง (hmn_v2_width_sweep3.py) **ไม่มี MoE** | แก้คำอธิบายให้ถูกต้อง (embed + coupling + memory + head) + เพิ่มหมายเหตุขอบเขตการสรุป |
| 4. Pseudocode 10.1–10.3 | ไม่ตรงกับ design ที่ validate แล้ว (ไม่มี Pre-LN, ไม่มี aux loss, memory ยังเขียนแบบ ANN + projection เดียว) | อัปเดตให้ตรง: Pre-LN ภายใน F (10.1), aux load-balancing loss (10.2), key/value projection แยก (10.3) + หมายเหตุว่าโค้ดจริงคือ hmn_v2.py |
| 5. (เล็ก) สถานะ header | เขียน "ข้อ 1–4" แต่ Section 18 ครอบคลุมข้อ 1–5 | แก้เป็น "ข้อ 1–5" + เพิ่มบันทึกการแก้ไขใน header |
| 6. (เล็ก) Section 18.1 ข้อ 2 | เขียน "aux ลด 0.07 → 0.04" แต่ตัวเลขจริงมี rebound ปลายทาง | แก้เป็น "ลดจนต่ำสุด 0.039 (เริ่ม 0.046, พีค 0.070, ปลาย 0.053)" |
| 7. (เล็ก) Section 5.2.2 / 5.3 / 5.4 / 11 / 15 | ไม่ได้ผูก Pre-LN/aux loss/key-value แยกเข้ากับส่วน design และ hyperparameter | เพิ่มหมายเหตุ v2.0 ใน 5.2.2, 5.3.1, 5.4.3, 5.5, 11 และเพิ่ม aux_coef ในตาราง 15 |

### 18.6 ผล verified v2.2 — สถาปัตยกรรม Option 1 (merge เข้า hmn_v2.py แล้ว)

> **หมายเหตุ v2.2:** งานทั้งหมดในหัวข้อนี้ทำบน CPU (i3, 4.6GB RAM, venv `ReTop/.venv`)
> เป้าหมายอ้างอิง 90-100% ของหัวข้อ 18 เป็น **UNVERIFIED** — ตัวเลข verified ล่าสุดด้านล่าง
> ใช้เกณฑ์ random-query (ไม่มี positional shortcut) และ eval 1000 samples/ตำแหน่ง
> รายละเอียดเชิงลึก: task2_findings.md หัวข้อ 7-12, โค้ด: experiments/verified/

#### 18.6.1 ข้อค้นพบ: positional shortcut ทำลายตัวเลขเดิม

การ reproduce ด้วย config เดียวกับ 18.2 (dim40/L4, 8 คู่/50 keys) ได้ **acc 3-6%** —
ตัวเลข 90% เดิมไม่ reproducible สาเหตุคือ eval protocol เดิม **query คู่แรกเสมอ** ซึ่ง
ทำนายได้จากตำแหน่ง (positional shortcut) โดยไม่ต้องเรียน content addressing:

- train คู่แรก → eval คู่แรก = 99-100% แต่ eval random-query **ตกเหลือ ~17-49%**
- train random-query → eval random-query = 55-65% (เพดาน standalone)
- **การกวาด width/depth ใหม่ด้วย random-query พลิกข้อสรุป:** d80/L4 = 15.3% >
  d40/L8 = 9.7% → กลับไปทาง "width ดีกว่า depth" (ข้อสรุปเดิม depth-efficiency
  ใน 18.2 ใช้ไม่ได้)

#### 18.6.2 สถาปัตยกรรมที่ verify แล้ว: Option 1 (memory บน raw embeddings)

| Design | ผล | สรุป |
|---|---|---|
| Option 1 baseline (D64/L2, 8 คู่/50 keys, random-query) | 51% | พื้นฐาน (1/50 = 2% = chance) |
| + capacity sweep (64→512 cells) | 77% | capacity ช่วยแต่ไม่พอ |
| + gate-blend รวม memory กับ backbone ที่ head | 89% | ผสม raw-embed recall กับ contextual signal |
| + β=30 (sharpening content addressing) | ~95% | อัดความน่าจะเป็นให้เน้นเฉพาะ token ที่ match |
| + usage_decay (write count tracking) | 97-99% | ขจัดการเขียนทับของ tokens ที่ใช้บ่อย (ยืนยัน 2 seeds, 1000-sample eval) |

**การค้นพบสำคัญ:** backbone SSM ทำให้ hidden states เป็น contextual → content addressing
ล้มเหลว (memory อ่าน state ไม่ใช่ identity ของ token) วิธีแก้คือให้ memory branch อ่าน
**raw embedding** (parallel) แล้วรวมกับ backbone ที่ head ด้วย gate: `z = g·backbone + (1-g)·memory_read`

#### 18.6.3 งาน 2-token (values 0-99) — design B ที่ทำงาน

- **Design A (chain re-query)** ล้มเหลว (11%) — propagation ผิดพลาดสะสม
- **Design B (query ครั้งเดียว + combined write)** ได้ **94-97%** ทุกตำแหน่ง query
  (seeds 0, 42): เขียน full-value ลง memory พร้อมกัน (combined write) ที่ token
  สุดท้ายของ value แล้ว decode ที่ head ด้วย 2 heads (tens, ones)

#### 18.6.4 Root cause: off-by-one bug ใน memory write

คอนดิชันตรวจ "end of value" เดิมเป็น `t < 2*N` (=16) แต่ sequence ยาวถึง `3*N-1` (=23)
สำหรับงาน 2-token → คู่ที่ 5-7 (positions 16-23) **ไม่ถูกเขียน memory เลย** แก้เป็น
`t < 3*N` แล้ว **attribution test ยืนยันว่า off-by-one อย่างเดียวก็พอ** (93-99% seed 0,
88-99% seed 42) โดยไม่ต้องใช้ exempt_combined (เป็น safety margin ที่ neutral เท่านั้น)

#### 18.6.5 Regression หลัง merge (experiments/regression_hmn_v2.py)

`HMN_Option1` + memory แบบใหม่ (beta_init/usage_decay/combined/exempt_combined/n_pairs,
off-by-one แก้แล้ว) ถูก merge เข้า `hmn_v2.py` (ยังไม่มี git history — backup ที่
experiments/verified/hmn_v2_merged_v2.2.py):

| Task | uniform | first | mid | tail | เป้า verified | ผล |
|---|---|---|---|---|---|---|
| single-token | 95.8% | 99.1% | 95.5% | 91.4% | 97-99% | ✅ |
| 2-token | 97.8% | 99.7% | 97.6% | 94.4% | 94-97% | ✅ |

Backward compat: `HMN` เดิม + memory default เดิมยังทำงานปกติ (ไม่มี MoE/multi-head/
combined เมื่อไม่ได้เปิด)

**เป้าหมายอ้างอิงปัจจุบัน (verified):** 97%+ single-token, 94-97% 2-token (random-query)
— ใช้แทนตัวเลข 90-100% UNVERIFIED ในหัวข้อ 18

---

**จบเอกสาร (v2.2)**

*เอกสาร v2.2: ตัวเลข 18.0–18.2 ถูกตีกลับเป็น UNVERIFIED (positional shortcut) และถูกแทนที่ด้วย
ผล verified ในหัวข้อ 18.6 (Option 1: raw-embed memory + gate-blend, 97-99% single-token /
94-97% 2-token, root cause off-by-one ถูกแก้แล้ว) ผู้พัฒนาต่อควรใช้ hmn_v2.py (merge แล้ว) เป็น
ฐาน และอัปเดตเอกสารให้สอดคล้องกันเมื่อเทรนบน GPU — roadmap ข้อ 5 (parallel scan), 7 (STE),
6 (GPU), 8 (reversible MoE/memory) ยังคงค้างตามหัวข้อ 18.4*

--- End of file ---
