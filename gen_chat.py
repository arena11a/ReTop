"""gen_chat — streaming chat-pairs dataset generator for retop training.

Produces .jsonl chat records (user/assistant, the format retop.py and
train_v3.py train on) up to a target token budget. Content is deterministic
from `--seed`, combinatorial over topic/entity pools (so it is varied rather
than N identical lines), and streamed — a large budget costs disk space, not
RAM.

  python gen_chat.py --domain english --level general --target-tokens 10_000_000 \
      --out data/english_10m.jsonl --seed 0
  python gen_chat.py --domain math --level advanced --target-tokens 120_000_000 \
      --out data/math_advanced_120m.jsonl --seed 0

Token budgeting: a "token" is approximated at ~4 characters per token (the
typical ratio for a BPE vocab ~4000 on English). The true count depends on the
tokenizer you train; the sidecar records both the char budget used and the
approx token estimate.

Levels:
  english: general  general knowledge / definitions / how-to / trivia dialogue
  math:    general  arithmetic, percentages, time, simple area
           complex  multi-step, fractions, ratios, linear equations, geometry
           advanced systems, quadratics, logs, exponents, trig, applied math
"""
import argparse
import json
import math
import os
import random

CHARS_PER_TOKEN = 4.0


def _yes(rng, p=0.5):
    return rng.random() < p


def _jump(rng, lo, hi):
    return rng.randint(lo, hi)


def _frac(rng):
    n = rng.randint(1, 12)
    d = rng.randint(2, 12)
    return n, d


# ---------------------------------------------------------------------------
# English: general knowledge / definitions / how-to / trivia
# ---------------------------------------------------------------------------

EN_SUBJECTS = [
    "photosynthesis", "gravity", "the water cycle", "black holes", "DNA",
    "electromagnetism", "climate change", "the Industrial Revolution",
    "batteries", "the solar system", "vaccination", "erosion", "sound waves",
    "friction", "evolution", "the periodic table", "ocean currents",
    "volcanoes", "the human heart", "light refraction",
]
EN_DEF = {
    "photosynthesis": "the process by which green plants use sunlight to turn carbon dioxide and water into glucose and oxygen",
    "gravity": "the force that attracts two bodies toward each other",
    "a black hole": "a region of spacetime where gravity is so strong that nothing, not even light, can escape",
    "DNA": "the molecule that carries the genetic instructions for living organisms",
    "electromagnetism": "the branch of physics that studies the interaction between electric currents and magnetic fields",
    "climate change": "the long-term shift in global temperatures and weather patterns",
    "a battery": "a device that stores chemical energy and converts it into electrical energy",
    "the solar system": "the sun together with the planets, moons, and other bodies that orbit it",
    "vaccination": "the administration of a vaccine to build immunity against a disease",
    "erosion": "the gradual wearing away of land by water, wind, or ice",
    "sound": "a vibration that travels through a medium as a wave of pressure",
    "friction": "the resistance that one surface or object encounters when moving over another",
    "evolution": "the change in the inherited traits of a population over generations",
    "ocean currents": "large-scale movements of seawater driven by wind and temperature differences",
    "a volcano": "an opening in the planet's crust through which molten rock and gas escape",
    "refraction": "the bending of a wave when it passes from one medium into another",
}
EN_HOWTO = [
    ("how do you {v}", "{c}."),
    ("what is the first step to {v}", "Start by {c}."),
    ("can you explain {t} in simple terms?", "{d}."),
]
EN_TRIVIA = [
    ("which planet is known as the red planet?", "Mars"),
    ("what gas do plants absorb from the air?", "carbon dioxide"),
    ("how many sides does a hexagon have?", "six"),
    ("what is the largest ocean on earth?", "the Pacific Ocean"),
    ("what part of the cell contains the genetic material?", "the nucleus"),
    ("what force keeps us on the ground?", "gravity"),
    ("which metal is liquid at room temperature?", "mercury"),
    ("what is the hardest natural substance on earth?", "diamond"),
    ("how many continents are there?", "seven"),
    ("what is H2O more commonly known as?", "water"),
]
EN_VERBS = ["boil an egg", "start a compost pile", "change a tire", "make tea",
            "grow tomatoes", "take a good photo", "save money", "learn a language",
            "plan a trip", "stay healthy"]
EN_STEPS = ["gathering what you need", "warming up first", "reading the manual",
            "making a list", "asking someone who knows", "starting small",
            "keeping a steady rhythm", "cleaning up as you go", "checking twice",
            "taking a short break"]
EN_ADV = [
    ("what is the main idea of {t}?",
     "It is best understood as a system with inputs, a process, and an output."),
    ("why is {t} important?",
     "Because it shapes how other parts of the natural world behave and connect."),
    ("compare {t} with its opposite.",
     "Both are extremes on a spectrum; the interesting behavior happens in the middle."),
]


def gen_english(rng):
    kind = rng.randrange(4)
    if kind == 0:
        topic = rng.choice(EN_SUBJECTS)
        if topic in EN_DEF:
            u = f"what is {topic}?"
            a = EN_DEF[topic] + "."
            return u, a
    if kind == 1:
        q, a = rng.choice(EN_TRIVIA)
        return q + "?", a
    if kind == 2:
        v = rng.choice(EN_VERBS)
        u = f"how do you {v}?"
        a = "Start by " + rng.choice(EN_STEPS) + "."
        return u, a
    t = rng.choice(EN_SUBJECTS)
    tpl = rng.choice(EN_ADV)
    return tpl[0].format(t=t), tpl[1].format(t=t)


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def gen_math_general(rng):
    kind = rng.randrange(6)
    if kind == 0:
        a, b = _jump(rng, 2, 120), _jump(rng, 2, 120)
        op = rng.choice(["+", "-", "*"])
        res = {"+": a + b, "-": a - b, "*": a * b}[op]
        return f"what is {a} {op} {b}?", f"{res}"
    if kind == 1:
        d = _jump(rng, 2, 40) * 2
        p = _jump(rng, 10, 200)
        res = d * p
        return (f"a train travels at {p} km/h for {d} hours, how far does it "
                f"go?"), f"{res} km"
    if kind == 2:
        total = _jump(rng, 4, 50) * 5
        each = total // 5
        return (f"five friends share {total} marbles equally, how many does "
                f"each get?"), f"{each}"
    if kind == 3:
        price = _jump(rng, 5, 80) * 5
        pct = rng.choice([10, 20, 25, 50])
        disc = price * pct // 100
        final = price - disc
        return (f"a jacket costs ${price} and is {pct}% off, what is the sale "
                f"price?"), f"${final}"
    if kind == 4:
        l, w = _jump(rng, 3, 25), _jump(rng, 3, 25)
        area = l * w
        return f"what is the area of a {l} by {w} rectangle?", f"{area}"
    a, b = _jump(rng, 30, 300), _jump(rng, 30, 300)
    res = a + b
    return (f"a book has {a} pages and you read {b} more, how many pages in "
            f"total?"), f"{res}"


def gen_math_complex(rng):
    kind = rng.randrange(6)
    if kind == 0:
        n, d = _frac(rng)
        a = _jump(rng, 1, 10)
        res = (a * d + n) / d
        return f"what is {a} and {n}/{d} as a single fraction?", f"{a * d + n}/{d}"
    if kind == 1:
        r1, r2 = _jump(rng, 2, 12), _jump(rng, 2, 12)
        res = r1 * r2
        return (f"a field is {r1} times as long as it is wide; its width "
                f"is {r2} m. what is the length in meters?"), f"{res} m"
    if kind == 2:
        x = _jump(rng, 2, 15)
        a = _jump(rng, 2, 10)
        b = a * x + _jump(rng, 1, 9)
        return f"solve for x: {a}x + {b - a * x} = {b}", f"x = {x}"
    if kind == 3:
        base = _jump(rng, 4, 40)
        h = _jump(rng, 3, 30)
        area = base * h // 2
        return (f"a triangle has base {base} cm and height {h} cm, what is its "
                f"area?"), f"{area} square cm"
    if kind == 4:
        n, d = _frac(rng)
        frac = n / d
        pct = round(frac * 100)
        return (f"what percent is {n}/{d}?"), f"{pct}%"
    n = _jump(rng, 2, 12)
    d = n * _jump(rng, 2, 6)
    g = math.gcd(n, d)
    return f"simplify {n}/{d}", f"{n // g}/{d // g}"


def gen_math_advanced(rng):
    kind = rng.randrange(7)
    if kind == 0:
        x = _jump(rng, 2, 9)
        a = _jump(rng, 2, 5)
        c = _jump(rng, 1, 10)
        b = a * x + c
        return f"solve for x: {a}x + {c} = {b}", f"x = {x}"
    if kind == 1:
        p = _jump(rng, 3, 12)
        k = _jump(rng, 1, 6)
        res = p ** k
        return f"what is {p}^{k}?", f"{res}"
    if kind == 2:
        n = rng.choice([25, 49, 64, 81, 100, 121, 144, 169, 196, 225])
        res = int(math.sqrt(n))
        return f"what is the square root of {n}?", f"{res}"
    if kind == 3:
        r = _jump(rng, 2, 12)
        a = 3 * r * r
        return f"a circle has radius {r}, what is its area (pi approx 3)?", f"{a}"
    if kind == 4:
        x = _jump(rng, 2, 8)
        c = x * x - 2 * x
        return f"solve x^2 - {2 * x}x + {c} = 0 (one repeated root)", f"x = {x}"
    if kind == 5:
        n = rng.choice([8, 27, 64, 125, 216, 343, 512, 729, 1000])
        res = round(n ** (1 / 3))
        return f"what is the cube root of {n}?", f"{res}"
    p = _jump(rng, 1000, 9000)
    r = rng.choice([2, 3, 5, 10])
    t = rng.choice([2, 3, 5])
    res = p * (100 + r) ** t // 100 ** t
    return (f"if ${p} grows {r}% each year, about how much after {t} years "
            f"(compound, integer)?"), f"about ${res}"


GENERATORS = {
    "english": gen_english,
    "math": gen_math_general,
    "math_general": gen_math_general,
    "math_complex": gen_math_complex,
    "math_advanced": gen_math_advanced,
}


def record(u, a):
    return {"messages": [
        {"role": "user", "content": u},
        {"role": "assistant", "content": a},
    ]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", required=True,
                    choices=["english", "math", "math_general", "math_complex", "math_advanced"])
    ap.add_argument("--target-tokens", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gen = GENERATORS[args.domain]
    target_chars = int(args.target_tokens * CHARS_PER_TOKEN)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    meta = {"generator": "gen_chat", "domain": args.domain, "seed": args.seed,
            "target_tokens": args.target_tokens, "chars_per_token": CHARS_PER_TOKEN}
    n = 0
    chars = 0
    t0 = os.times() if hasattr(os, "times") else None
    with open(args.out, "w", encoding="utf-8") as f:
        while chars < target_chars:
            u, a = gen(rng)
            line = json.dumps(record(u, a), ensure_ascii=False) + "\n"
            f.write(line)
            n += 1
            chars += len(line)
            if not args.quiet and n % 50000 == 0:
                print(f"  {chars/1e6:.1f}/{target_chars/1e6:.1f}M chars "
                      f"({n:,} records)", flush=True)
    meta.update(records=n, chars=chars,
                approx_tokens=int(chars / CHARS_PER_TOKEN))
    with open(args.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{args.out}: {n:,} records, {chars/1e6:.1f}M chars, "
          f"approx {meta['approx_tokens']/1e6:.1f}M tokens")


if __name__ == "__main__":
    main()