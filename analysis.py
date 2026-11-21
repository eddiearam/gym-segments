"""Gym member segments: who should get the monthly offer?

k-means clustering of gym members into segments, used to decide which segments
get the $10 extras credit. Every number on the slides is written to results.txt
by this script, and every chart is saved to this folder.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

SEED = 42
TEST_SHARE = 0.30
K_RANGE = range(1, 9)
K_CHOSEN = 4  # elbow of the training curve, confirmed on the test curve (section 3)

# Finance's working assumptions, applied member by member
CREDIT_COST = 10        # every member who gets the credit uses it
SPEND_THRESHOLD = 20    # members spending this much or more would spend the same anyway
EXTRA_SPEND = 25        # members below the threshold spend this much more with the credit
MARGIN = 0.60           # margin on extras

# member_id is an identifier, not a description of the member, so it is left out
FEATURES = ["age", "visits_per_month", "avg_visit_minutes", "classes_per_month",
            "months_member", "monthly_extras_spend"]

lines = []
def log(s=""):
    print(s)
    lines.append(s)

def credit_net(spend):
    """Gym's monthly net from giving one member the credit (finance's assumptions)."""
    spend = np.asarray(spend)
    return np.where(spend >= SPEND_THRESHOLD, -CREDIT_COST, EXTRA_SPEND * MARGIN - CREDIT_COST)

# ---------------------------------------------------------------- 1. Data and split
df = pd.read_csv("gym_members.csv")
log("=== DATA ===")
log(f"gym_members.csv: {len(df)} members, {df.shape[1]} columns, averages over the last 3 months")
log(f"Average extras spend: ${df.monthly_extras_spend.mean():.2f} a month")
train, test = train_test_split(df, test_size=TEST_SHARE, random_state=SEED)
log(f"Random 70/30 split (seed={SEED}): train = {len(train)}, test = {len(test)}")
log(f"Features: {FEATURES}")
log("Left out: member_id (an identifier, says nothing about the member)")
log("Scaling: each feature standardised (mean 0, sd 1) using training members only")
log()

scaler = StandardScaler().fit(train[FEATURES])
X_train = scaler.transform(train[FEATURES])
X_test = scaler.transform(test[FEATURES])

def spread(X, centers):
    """Sum of squared distances from each member to its nearest center."""
    d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return d.min(axis=1).sum()

# ---------------------------------------------------------------- 2-3. k = 1..8, train and test spread
log("=== WITHIN-GROUP SPREAD BY k (centers fitted on training members only) ===")
log(f"{'k':>3}{'Train spread':>15}{'Test spread':>14}{'Test drop vs k-1':>19}")
models, train_spread, test_spread = {}, [], []
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_train)
    models[k] = km
    train_spread.append(km.inertia_)
    test_spread.append(spread(X_test, km.cluster_centers_))
    drop = "" if k == 1 else f"{1 - test_spread[-1] / test_spread[-2]:.0%}"
    log(f"{k:>3}{train_spread[-1]:>15.1f}{test_spread[-1]:>14.1f}{drop:>19}")
base_test, v1_test = test_spread[0], test_spread[K_CHOSEN - 1]
log(f"Chosen k = {K_CHOSEN}: last big drop on both curves; after it each extra group cuts spread by under 10%")
log(f"Test spread: baseline (k=1) {base_test:.1f}  v1 (k={K_CHOSEN}) {v1_test:.1f}  "
    f"-> {1 - v1_test / base_test:.0%} lower")
log()

v1 = models[K_CHOSEN]

# ---------------------------------------------------------------- 4. Segments
train = train.assign(segment=v1.labels_)
centers = pd.DataFrame(scaler.inverse_transform(v1.cluster_centers_), columns=FEATURES)

# Name each segment from its center (the average member)
def name_segment(c):
    if c.months_member < 10 and c.visits_per_month < 6:
        return "New and testing"
    if c.age >= 55:
        return "Loyal seniors"
    if c.classes_per_month >= 4:
        return "Class lovers"
    return "Gym regulars"

names = {i: name_segment(centers.loc[i]) for i in centers.index}
assert len(set(names.values())) == K_CHOSEN, "segment names must be unique"

# Credit decision: give the credit to a segment only if, member by member,
# it makes the gym money on that segment's training members.
train["credit_net"] = credit_net(train.monthly_extras_spend)
seg_net = train.groupby("segment")["credit_net"].sum()
gets_credit = {i: bool(seg_net[i] > 0) for i in names}

log(f"=== v1 SEGMENTS (k = {K_CHOSEN}): centers = average member, training members ===")
summary = centers.round(1).copy()
summary.insert(0, "name", [names[i] for i in summary.index])
summary["members_train"] = train.segment.value_counts().sort_index()
summary["share_spend_under_20"] = train.groupby("segment").monthly_extras_spend.apply(
    lambda s: (s < SPEND_THRESHOLD).mean()).round(2)
summary["credit_net_train"] = seg_net.astype(int)
summary["gets_credit"] = [gets_credit[i] for i in summary.index]
log(summary.T.to_string())
log()

# ---------------------------------------------------------------- 5. Decision function
def assign(member: dict):
    """Return (segment name, whether this member gets the $10 credit)."""
    x = scaler.transform(pd.DataFrame([member])[FEATURES])
    seg = int(v1.predict(x)[0])
    return names[seg], gets_credit[seg]

new_member = dict(age=30, visits_per_month=4, avg_visit_minutes=42, classes_per_month=1,
                  months_member=5, monthly_extras_spend=10)
seg_name, credit = assign(new_member)
log("=== DECISION ===")
log("Credit goes to: " + ", ".join(names[i] for i in names if gets_credit[i]))
log("No credit for: " + ", ".join(names[i] for i in names if not gets_credit[i]))
log(f"New member {new_member}")
log(f"  -> segment: {seg_name}, gets credit: {'yes' if credit else 'no'}")
log()

# ---------------------------------------------------------------- 6. Value, all 330 members
df["segment"] = v1.predict(scaler.transform(df[FEATURES]))
df["segment_name"] = df.segment.map(names)
df["credit_net"] = credit_net(df.monthly_extras_spend)
df["v1_credit"] = df.segment.map(gets_credit)
base_total = df.credit_net.sum()
v1_total = df.loc[df.v1_credit, "credit_net"].sum()

log(f"=== VALUE: gym's monthly net from the credit, all {len(df)} members, member by member ===")
log(f"{'Segment':<18}{'Members':>8}{'Avg spend':>11}{'Under $20':>11}{'Net if credited':>17}{'Credit':>8}")
for i in names:
    s = df[df.segment == i]
    log(f"{names[i]:<18}{len(s):>8}{s.monthly_extras_spend.mean():>11.2f}{(s.monthly_extras_spend < SPEND_THRESHOLD).sum():>11}"
        f"{s.credit_net.sum():>17,.0f}{'yes' if gets_credit[i] else 'no':>8}")
log()
def usd(v):
    return f"{'-' if v < 0 else '+'}${abs(v):,.0f}"
log(f"Baseline (credit to all {len(df)}):   {usd(base_total)} a month   ({usd(base_total * 12)} a year)")
log(f"v1 (credit to {int(df.v1_credit.sum())} members):     {usd(v1_total)} a month   ({usd(v1_total * 12)} a year)")
log(f"Difference (v1 - baseline):  {usd(v1_total - base_total)} a month   ({usd((v1_total - base_total) * 12)} a year)")
log(f"Credits sent: baseline {len(df)}, v1 {int(df.v1_credit.sum())}")

with open("results.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
df.to_csv("member_segments.csv", index=False)

# ---------------------------------------------------------------- 7. Charts
fig, ax = plt.subplots(figsize=(7, 4.3))
ks = list(K_RANGE)
ax.plot(ks, train_spread, marker="o", color="#1F5AA6", label=f"Training members ({len(train)})")
ax.plot(ks, test_spread, marker="o", color="#E07B00", label=f"Test members ({len(test)}), training centers")
ax.axvline(K_CHOSEN, color="grey", ls="--", lw=1)
ax.text(K_CHOSEN + 0.1, max(train_spread) * 0.92, f"elbow: k = {K_CHOSEN}", color="grey")
ax.set_xticks(ks)
ax.set_xlabel("Number of groups (k)")
ax.set_ylabel("Within-group spread\n(sum of squared distances)")
ax.set_title("Within-group spread by number of groups")
ax.legend(frameon=False)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("elbow.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4.2))
vals = [base_total, v1_total]
bars = ax.bar(["Baseline:\ncredit to everyone", f"v1: credit to\n{', '.join(names[i] for i in names if gets_credit[i])}"],
              vals, color=["#B0B0B0", "#1F5AA6"], width=0.55)
for b, v in zip(bars, vals):
    off = 25 if v >= 0 else -25
    ax.text(b.get_x() + b.get_width() / 2, v + off, f"{'-' if v < 0 else '+'}${abs(v):,.0f} a month",
            ha="center", va="bottom" if v >= 0 else "top", fontsize=11)
ax.axhline(0, color="black", lw=0.8)
lo, hi = min(vals + [0]), max(vals + [0])
ax.set_ylim(lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo))
ax.set_ylabel("Gym's monthly net from the credit ($)")
ax.set_title(f"Monthly net from the $10 credit, all {len(df)} members")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("value.png", dpi=200)
plt.close(fig)
