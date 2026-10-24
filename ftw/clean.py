"""Turn any state (clean live data or a dirty practice market) into the typed fields the brain reads."""
from .game import num


def _d(x):
    return x if isinstance(x, dict) else {}


def _l(x):
    return x if isinstance(x, list) else []


def _s(x):
    return x if isinstance(x, str) else None


def sanitize(state):
    st = _d(state)
    rules = _d(st.get("rules"))
    spots = []
    for s in _l(rules.get("spots")):
        if isinstance(s, str) and s not in spots:
            spots.append(s)
    mc = {}
    for a, row in _d(rules.get("move_costs")).items():
        mc[a] = {b: max(0.0, num(v, 0.0)) for b, v in _d(row).items()}
    fc = _d(st.get("forecast"))
    events = []
    for e in _l(fc.get("events")):
        if isinstance(e, dict):
            events.append({"key": _s(e.get("key")) or "", "effect": str(e.get("effect") or ""),
                           "spot": _s(e.get("spot")), "opponent": _s(e.get("opponent"))})
    you_raw = st.get("you")
    you = None
    if isinstance(you_raw, dict) and _s(you_raw.get("id")):
        staff = _d(you_raw.get("staff"))
        you = {"id": you_raw["id"],
               "reputation": {k: min(100.0, max(0.0, num(v, 50.0))) for k, v in _d(you_raw.get("reputation")).items()},
               "last_spot": _s(you_raw.get("last_spot")),
               "staff": {"employees": min(20.0, max(0.0, num(staff.get("employees"), 4.0))),
                         "wage": max(0.0, num(staff.get("wage"), 0.0)),
                         "capacity": min(1000.0, max(0.0, num(staff.get("capacity"), 0.0)))}}
    ld_raw = st.get("last_day")
    last_day = None
    if isinstance(ld_raw, dict):
        sp = {}
        for s, v in _d(ld_raw.get("spots")).items():
            rows, seen = [], set()
            for t in _l(_d(v).get("trucks")):
                if isinstance(t, dict) and _s(t.get("id")) and t["id"] not in seen:
                    seen.add(t["id"])
                    rows.append({"id": t["id"], "price": min(30.0, max(1.0, num(t.get("price"), 10.0))),
                                 "quality": _s(t.get("quality")), "meals": min(1000.0, max(0.0, num(t.get("meals"), 0.0))),
                                 "sold": max(0.0, num(t.get("sold"), 0.0))})
            sp[s] = {"trucks": rows}
        last_day = {"day": ld_raw.get("day") if isinstance(ld_raw.get("day"), int) else None, "spots": sp}
    trucks, seen = [], set()
    for t in _l(st.get("trucks")):
        if isinstance(t, dict) and _s(t.get("id")) and t["id"] not in seen:
            seen.add(t["id"])
            trucks.append({"id": t["id"], "employees": t.get("employees") if isinstance(t.get("employees"), (int, float))
                           and not isinstance(t.get("employees"), bool) else None,
                           "reputation": {k: min(100.0, max(0.0, num(v, 50.0)))
                                          for k, v in _d(t.get("reputation")).items()}})
    info = {}
    for s, v in _d(st.get("spot_info")).items():
        if isinstance(v, dict):
            info[s] = {"x": num(v.get("x")), "z": num(v.get("z"))}
    laws = [l for l in _l(_d(st.get("politics")).get("laws")) if isinstance(l, dict)]
    return {
        "mode": _s(st.get("mode")), "market": st.get("market") if isinstance(st.get("market"), (int, str)) else None,
        "market_id": st.get("market_id") if isinstance(st.get("market_id"), (int, str)) else None,
        "days": int(num(st.get("days"), 30)), "order_for_day": int(num(st.get("order_for_day"), 0)),
        "orders_open": st.get("orders_open", True) is not False,
        "rules": {"spots": spots, "move_costs": mc},
        "features": {k: v is True for k, v in _d(st.get("features")).items()},
        "forecast": {"weather": {k: max(0.0, num(v, 0.0)) for k, v in _d(fc.get("weather")).items()},
                     "events": events,
                     "expected_customers": {k: min(5000.0, max(0.0, num(v, 0.0))) for k, v in _d(fc.get("expected_customers")).items()},
                     "planned": ({k: int(min(50, max(0, num(v, 0)))) for k, v in fc["planned"].items()}
                                 if isinstance(fc.get("planned"), dict) else None)},
        "spot_info": info, "politics": {"laws": laws}, "you": you, "last_day": last_day, "trucks": trucks,
        "_peek": st.get("_peek") if isinstance(st.get("_peek"), dict) else None,
        "_posted": _s(st.get("_posted")),     # the spot of the order we already have in tonight (runner)
    }
