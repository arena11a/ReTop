# Task 1 — Arithmetic Curriculum: Findings Report (v2)

**คำถามที่ต้องการตอบ (roadmap ข้อ 6):** Generation คณิตศาสตร์ที่ผิด (5+7 → 5+7*77)
มาจากข้อจำกัดของ HMN architecture หรือขนาดโมเดล/data?

## สรุปผลชี้ขาด

### 1. ปัญหาแรกที่เจอ: BPE tokenizer ทำลาย place-value structure
- Tokenizer ปัจจุบัน (BPE, vocab 3190) เข้ารหัสเลขไม่สม่ำเสมอ: `57` → 1 token [2126]
  แต่ `357` → 2 tokens [24, 2126], `100` → [22, 869]
- โมเดลต้องเรียน mapping จาก sequence ไปยัง output token ที่ไม่สัมพันธ์กับโครงสร้างตัวเลข
  → ต้องแก้ representation เป็น per-digit tokenization ก่อน

### 2. Per-digit tokenization: HMN memorize ได้ แต่ generalize ข้าม digit-length ไม่ได้
ทดสอบ HMN v2 (251K params) บน per-digit encoding:

| การทดลอง | Train 1-digit | Val 2-digit |
|---|---|---|
| loss เฉพาะ answer token | 1.0 (memorize 100%) | 0.0 |
| train ครบ 55 คู่หลักเดียว | 0.99 | 0.0 |
| mixed curriculum A+B | 0.0 (สับสน) | 0.03 |

### 3. Dense Transformer เป็น baseline: ได้ผลเหมือนกัน (สำคัญที่สุด)
| โมเดล | params | train 1-digit | generalize 2-digit |
|---|---|---|---|
| HMN v2 | 251K | 1.0 | 0.0 |
| Transformer dim64/L2 | 384K | 1.0 | 0.0 |
| Transformer dim128/L3 | 1.0M | 1.0 | 0.0 |

## บทสรุป
**ข้อจำกัด generation ไม่ใช่ข้อจำกัดของ HMN architecture** — Dense Transformer ที่ใหญ่กว่า
HMN ถึง 4 เท่าได้ผลเหมือนกัน (memorize 1-digit 100% แต่ generalize 2-digit 0%)
โมเดลเรียนรู้ "look-up table" ไม่ใช่ "operation" เมื่อเห็นเฉพาะหลักเดียว

สาเหตุเชิงลึก:
1. **ข้อมูลฝึกไม่พอ** — การเห็น 1-digit add ไม่ได้สอน carry/algorithm เลย
2. **ขนาดโมเดล** — 251K-1M params เล็กเกินไปจะเรียน compositional arithmetic algorithm
3. **โครงสร้างโจทย์** — ต้องมี curriculum หลายความยากผสม (Stage B/C/D) ตั้งแต่แรก

## ข้อเสนอการแก้
1. สร้าง curriculum dataset ให้เห็นความยากหลากหลายตั้งแต่ต้น (ไม่ใช่ train หลักเดียวแล้ว test 2 หลัก)
2. ใช้ per-digit tokenization (อย่าใช้ BPE ที่รวมเลข)
3. เพิ่มขนาดโมเดล (dim 96-128, L 6-8) เทรน Stage A+B+C+D
4. ตรวจผ่าน task accuracy ไม่ใช่แค่ loss

## ไฟล์ที่สร้าง
- `gen_arithmetic.py` — generator streaming/chunked ครบ (Stage A-D)
- `test_arithmetic.py` — unit tests (ผ่านทั้งหมด)
- `hmn_data/arithmetic/stage{A,B}_{train,val}/` — ชุดข้อมูล Stage A-B (10K train, 1K-27 val)
- `hmn_v2.py` — สถาปัตยกรรม rebuild จากเอกสาร v2.1
- `train_arithmetic.py` — training script พร้อม streaming dataset

## สถิติ dataset ที่สร้าง
- Stage A: 10K train (28 unique pairs), 27 val, ~6.4 tokens/problem, 1.4MB
- Stage B: 10K train (5391 unique), 1000 val (disjoint 100%), ~9.2 tokens, 1.4MB
- Generation time: <1s ต่อ stage, chunked 10K/ไฟล์ — ไม่เป็นคอขวด
