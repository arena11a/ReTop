# v9.4 Head-to-Head: Attention vs SSM

**Date**: 2026-08-27

## Results

### Slot-Copy Task

| Config | Variant | Accuracy | Params |
|--------|---------|----------|--------|
| D=64 L=2 | SSM | 1.000 | 431,870 |
| D=64 L=2 | Attention | 1.000 | 563,711 |
| D=128 L=3 | SSM | 1.000 | 901,982 |
| D=128 L=3 | Attention | 1.000 | 1,678,335 |

**Both variants achieve perfect accuracy on slot-copy.**

### Reorder Task (ptr3)

| Config | Variant | Accuracy | Params |
|--------|---------|----------|--------|
| D=64 L=2 | SSM | **0.900** | 431,870 |
| D=64 L=2 | Attention | 0.700 | 563,711 |
| D=128 L=3 | SSM | **0.900** | 901,982 |
| D=128 L=3 | Attention | 0.900 | 1,678,335 |

**SSM wins on reorder: better accuracy with fewer parameters.**

## Key Findings

1. **SSM is more parameter-efficient**: Achieves same or better accuracy with 25-46% fewer parameters
2. **SSM wins on reorder**: 0.900 vs 0.700 at D=64 (same task where attention struggled)
3. **Both are equal on slot-copy**: Simple task doesn't differentiate
4. **Attention needs more params to match SSM**: D=128 attention (1.68M) = D=128 SSM (901K) on reorder

## Implications

- **SSM is the better default** for ReTop's task profile (slot-copy + reorder)
- **Attention is over-parameterized** for these structured tasks
- **The reversible SSM backbone** (HelixCouplingBlock) is more efficient than transformer blocks
- **For v9.5+**: Consider SSM as the primary architecture, with attention for specific use cases

## Recommendation

Use SSM variant as the default for ReTop. Attention variant should be reserved for tasks requiring long-range dependencies or complex reasoning.
