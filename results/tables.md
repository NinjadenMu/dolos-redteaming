# Paper tables

## Table 1: Selective dependence on label

| Label | Dolos JS [95% interval] | Qwen JS [95% interval] |
|---|---:|---:|
| API_KEY | 0.4320 [0.3629, 0.4860] | 0.0005 [0.0002, 0.0011] |
| password | 0.4243 [0.3397, 0.4924] | 0.0007 [0.0002, 0.0013] |
| access_token | 0.4202 [0.3423, 0.4801] | 0.0004 [0.0002, 0.0009] |
| Unlabeled | 0.4358 [0.3489, 0.4920] | 0.0008 [0.0003, 0.0015] |
| random_string | 0.0757 [0.0155, 0.1360] | 0.0005 [0.0002, 0.0009] |
| record_id | 0.0003 [0.0001, 0.0005] | 0.0006 [0.0002, 0.0012] |
| session_id | 0.0003 [0.0001, 0.0006] | 0.0005 [0.0002, 0.0010] |
| checksum | 0.0002 [0.0001, 0.0004] | 0.0007 [0.0002, 0.0013] |

## Table 3: Credential sensitivity after projection

| Label | Unmodified Dolos JS | Projected Dolos JS | Qwen JS |
|---|---:|---:|---:|
| API_KEY | 0.4320 [0.3629, 0.4860] | 0.0003 [0.0000, 0.0006] | 0.0005 [0.0002, 0.0011] |
| password | 0.4243 [0.3397, 0.4924] | 0.0002 [0.0000, 0.0004] | 0.0007 [0.0002, 0.0013] |
| access_token | 0.4202 [0.3423, 0.4801] | 0.0002 [0.0000, 0.0005] | 0.0004 [0.0002, 0.0009] |

## Table 4: Sequential secret reconstruction

| Condition | Characters recovered | Accuracy [95% interval] | Relative reduction |
|---|---:|---:|---:|
| Unmodified Dolos | 234/256 | 91.4% [85.9%, 95.3%] | — |
| Decimal projection, rank 8 | 43/217 | 19.8% [15.6%, 24.7%] | 78.3% |
| Hexadecimal projection, rank 16 | 17/234 | 7.3% [4.2%, 9.8%] | 92.1% |

## Capability suite

| Condition | arithmetic | lookup | copy | sort | comparison | Total |
|---|---:|---:|---:|---:|---:|---:|
| unmodified | 23/24 | 24/24 | 24/24 | 0/12 | 12/12 | 83/96 |
| decimal_rank8 | 23/24 | 24/24 | 24/24 | 0/12 | 12/12 | 83/96 |
| hexadecimal_rank16 | 23/24 | 24/24 | 24/24 | 0/12 | 12/12 | 83/96 |
