"""The rivals we predict: one line each. Adding a rival is one line here (plus, optionally, a port in ftw/adapters/).

name: (truck id, GitHub repo, entry script, entry args, port file in ftw/adapters/ or None, port kwargs, port seeds)

The repo's real bot runs under ftw/shim.py (rivalhost.py) from rivals_live/ (fetched by run.sh) or the vendored
rivals/ snapshot; the port, if any, runs in a worker process (rivals.py). A rival with neither is predicted by
persistence like any unknown truck.
"""
RIVALS = {
    "ibe": ("t_e2de7d", "Akshit-Jalli/foodtruck", "supervise.py", [], "Akshit-Jalli__foodtruck.py", {"restarts": []}, [0]),
    "wagyu": ("t_96a96a", "jtanSE/wagyuexpress", "run_bot.py", [], "jtanSE__wagyuexpress.py", {}, [0]),
    "cheapest": ("t_5f2743", "nicholasravakhah/5guysburgers", "bot.py", [], "nicholasravakhah__5guysburgers.py", {},
                 [0, 1, 2]),
    "bigbuck": ("t_876b0a", "Peezy0/nightshift-food-truck", "bot.py", [], "Peezy0__nightshift-food-truck.py",
                {"legacy_through": 0}, [0]),
    # T.H.E @ d3b4b8d0 (intel policy.py): `bot.py --profile final` after a restart (final mode roams all 6 spots).
    # 12 scenarios is their final-mode floor and 0.6-0.7 s/night on the idle PowerSpec (8 is slower: 0.85 s);
    # elapsed 0.6 pins their wall-clock scenario resize. The old 5718151 port (siyanshvir__TruckHomeEatsBot.py)
    # stays the rehearsal sim's T.H.E (harness/lineup.py).
    "the": ("t_d64e26", "siyanshvir/TruckHomeEatsBot", "bot.py", [], "siyanshvir__TruckHomeEatsBot_final.py",
            {"samples": 12, "elapsed": 0.6}, [0]),
}

# Other candidate ports for a rival: the predictor runs them too and, each night, uses the candidate (primary
# first) with the best recent hit rate on that rival's actual orders. name: [(port file, kwargs, intel file)]
ALT_PORTS = {
    # T.H.E if they never restart: the still-running Oval-locked build 4468b94 (intel policy_4468b94.py)
    "the": [("siyanshvir__TruckHomeEatsBot_oval.py", {"samples": 12, "elapsed": 0.6}, "policy_4468b94.py")],
}


def snap(repo):
    """rivals/ folder name for a GitHub repo."""
    return repo.replace("/", "__", 1)
