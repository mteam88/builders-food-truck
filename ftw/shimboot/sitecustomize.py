"""Loaded by any Python a shimmed rival starts (PYTHONPATH points here): re-applies ftw/shim.py."""
import importlib.util as _u
import os as _os

_p = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "shim.py")
_s = _u.spec_from_file_location("_ftw_shim", _p)
_m = _u.module_from_spec(_s)
_s.loader.exec_module(_m)
if not _m.install_from_env():
    raise SystemExit("ftw shim: FTW_SHIM_REDIRECT missing in a shimmed child; refusing to run")
