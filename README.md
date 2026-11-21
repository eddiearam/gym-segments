# Gym member segments: who should get the monthly offer?

k-means clustering of gym members into segments, used to decide which segments get the $10 extras credit.

## Run it

```bash
pip install -r requirements.txt
python analysis.py
```

Every number and chart on the slides is written to this folder by that one script.

## Files

| Path | What it is |
|---|---|
| `analysis.py` | The full analysis: split, scaling, k-means for k = 1 to 8, test spread, segments, decision function, value, charts |
| `gym_members.csv` | 330 members, averages over the last 3 months |
| `results.txt` | Every number on the slides |
| `elbow.png` | Within-group spread by k, training and test members |
| `value.png` | Gym's monthly net from the credit, baseline vs v1 |
| `member_segments.csv` | Segment, credit decision and net for every member |

## Method

1. **Split:** random 70/30 split (seed 42): 231 training members, 99 test.
2. **Fit:** features are age, visits per month, minutes per visit, classes per month, months a member and monthly extras spend; `member_id` is left out. Each feature is standardised using the training members only. k-means (10 starts, seed 42) for k = 1 to 8 on the training members.
3. **Evaluate:** for each k, test members are assigned to the nearest training center without refitting, and the squared distances summed. Both curves bend at k = 4, so v1 has four segments. Test spread falls from 600.7 (k = 1, baseline) to 150.0 (k = 4).
4. **Decide:** a segment gets the credit only if crediting its training members makes the gym money under finance's assumptions, member by member. `assign(member)` returns the segment and whether that member gets the credit.
5. **Value:** finance's assumptions applied to every one of the 330 members: -$10 for a member spending $20 or more, +$5 for a member spending under $20.

## Key results

| Segment | Members (all 330) | Avg extras spend | Credit |
|---|---|---|---|
| New and testing | 70 | $9.30 | yes |
| Loyal seniors | 60 | $16.35 | no |
| Gym regulars | 109 | $24.60 | no |
| Class lovers | 91 | $46.93 | no |

| Plan | Monthly net from the credit |
|---|---|
| Baseline: credit to all 330 | -$1,200 |
| v1: credit to New and testing (70) | +$185 |

New member (30, 4 visits, 42 min, 1 class, 5 months, $10 extras): New and testing, gets the credit.
