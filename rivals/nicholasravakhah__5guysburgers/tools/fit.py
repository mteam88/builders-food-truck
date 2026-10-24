"""Fit the customer model (per-spot lognormal price limits and minimum-quality mix)
to logged days. Offline analysis only: needs numpy + scipy, the bot does not.

    python3 tools/fit.py state/calib.jsonl [state/live_days.jsonl ...]

Model: a customer's limit L ~ LogNormal(log(median), sigma), independent of their
minimum quality class c in {cheap, standard, premium} with probabilities pi.
For a spot-day where no truck sold out, the number served is
Binomial(customers, sum_c pi_c * P(L >= cheapest acceptable price for c)).
Single-truck sellouts are right-censored. Event days are fitted as a budget
multiplier plus a separate mix per (event, spot).
"""
import json
import math
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom, norm

QI = {"cheap": 0, "standard": 1, "premium": 2}
V = {"cheap": 8, "standard": 12, "premium": 16}


def load(paths):
    days = []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            ld = r["last_day"]
            reps = {t["id"]: t.get("reputation") or {} for t in (r.get("trucks_before") or [])}
            if not reps and r.get("trucks"):
                # live log: reputation after the day; undo the day's change approximately
                reps = {t["id"]: t.get("reputation") or {} for t in r["trucks"]}
            evk = {e.get("key") or e.get("name") for e in ld.get("events", [])}
            evs = {e.get("spot"): (e.get("key") or e.get("name")) for e in ld.get("events", [])}
            for s, v in ld["spots"].items():
                trucks = []
                sold_out = False
                for t in v["trucks"]:
                    rep = reps.get(t["id"], {}).get(s, 50)
                    trucks.append((QI[t["quality"]], float(t["price"]), V[t["quality"]] * (0.7 + 0.006 * rep) - t["price"],
                                   t["meals"], t["sold"]))
                    if t["meals"] > 0 and t["sold"] >= t["meals"]:
                        sold_out = True
                if not trucks or v["customers"] <= 0:
                    continue
                days.append(dict(spot=s, N=v["customers"], H=v["hungry"], trucks=trucks, sold_out=sold_out,
                                 event=evs.get(s), global_ev=evk, weather=ld.get("weather")))
    return days


def tag_of(d):
    if d["event"]:
        return d["event"]
    if "gameday" in d["global_ev"]:
        return "gameday-elsewhere"
    if "latenight" in d["global_ev"]:
        return "latenight-elsewhere"
    return "base"


def p_served(d, mu, sigma, pi):
    tot = 0.0
    for c in range(3):
        prices = [p for (qi, p, val, m, s) in d["trucks"] if qi >= c and val >= 0 and m > 0]
        if prices:
            tot += pi[c] * norm.sf((math.log(min(prices)) - mu) / sigma)
    return min(max(tot, 1e-9), 1 - 1e-9)


def nll(params, days):
    mu, ls, a, b = params
    sigma = math.exp(ls)
    e = np.exp([0.0, a, b])
    pi = e / e.sum()
    out = 0.0
    for d in days:
        pr = p_served(d, mu, sigma, pi)
        served = d["N"] - d["H"]
        if d["sold_out"]:
            if len(d["trucks"]) == 1:
                out -= math.log(max(binom.sf(served - 1, d["N"], pr), 1e-300))
            continue
        out -= binom.logpmf(served, d["N"], pr)
    return out


def fit(days):
    best = None
    for x0 in ([math.log(12), math.log(0.3), 0.0, -1.0], [math.log(11), math.log(0.2), 0.5, -0.5]):
        r = minimize(nll, x0, args=(days,), method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    mu, ls, a, b = best.x
    e = np.exp([0.0, a, b])
    pi = e / e.sum()
    return dict(median=round(math.exp(mu), 3), sigma=round(math.exp(ls), 4), mix=[round(x, 4) for x in pi],
                n=len(days), nll=round(best.fun, 1))


def main():
    days = load(sys.argv[1:])
    groups = defaultdict(list)
    for d in days:
        groups[(d["spot"], tag_of(d))].append(d)
    res = {}
    for k in sorted(groups):
        g = groups[k]
        usable = [d for d in g if not d["sold_out"] or len(d["trucks"]) == 1]
        if len(usable) < 4:
            print(k, "too few", len(usable))
            continue
        f = fit(usable)
        res["%s|%s" % k] = f
        print(k, f)
    json.dump(res, open("state/fit_result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
