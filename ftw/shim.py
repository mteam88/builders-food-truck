"""Run a rival's own entrypoint with every outbound connection sent to one loopback port.

    python -I ftw/shim.py --redirect 127.0.0.1:PORT [--via-unix SOCK] -- <entry.py> [args...]

Before the entry script runs (runpy, as __main__, with sys.argv and sys.path[0] set the way
`python entry.py` would), this module patches the process so that:

- socket.getaddrinfo answers every hostname with 127.0.0.1:PORT, socket.create_connection and
  socket.socket.connect/connect_ex rewrite every INET/INET6 destination to 127.0.0.1:PORT, and
  sendto/sendmsg to a non-loopback address is refused. A final check refuses any non-loopback
  destination that still reaches connect.
- https becomes plain http: http.client.HTTPSConnection.connect is HTTPConnection.connect, and
  ssl.SSLContext.wrap_socket hands back the plain socket. So urllib/http.client calls to
  https://<anything> land on the plain-http fake server on PORT.
- Child processes stay shimmed: the env carries FTW_SHIM_REDIRECT and PYTHONPATH=ftw/shimboot (whose
  sitecustomize.py calls install() again in any child Python). subprocess/os.exec*/os.spawn*/
  os.posix_spawn*/os.system may only start a Python interpreter; -I/-E (which would skip
  sitecustomize) are stripped from its argv, and the shim variables are forced into its env. Anything
  else (git, curl, sh) raises PermissionError.

--via-unix SOCK is for running inside `unshare -rn` (a private network namespace with no route out):
the shim brings up that namespace's lo, listens on 127.0.0.1:PORT there, and relays to the fake server's
Unix socket SOCK. Then even code that slips past the patches (ctypes, _socket, a non-Python binary)
reaches nothing. Without it, the patches are an in-process guard against accidents and casual escapes,
not a jail.
"""
import os
import socket
import sys

ENV_REDIRECT = "FTW_SHIM_REDIRECT"
BOOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shimboot")
_installed = None


def _is_loopback(host):
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    h = str(host).strip("[]").split("%")[0].lower()
    if h in ("localhost", "::1") or h.startswith("127.") or h == "::ffff:127.0.0.1":
        return True
    return False


def _is_python(path):
    b = os.path.basename(str(path)).lower()
    if b.startswith("python") or b.startswith("pypy"):
        return True
    try:
        return os.path.realpath(str(path)) == os.path.realpath(sys.executable)
    except (OSError, ValueError):
        return False


def install(host, port):
    """Patch this process. Idempotent."""
    global _installed
    if _installed:
        return
    port = int(port)
    host = str(host)
    if not _is_loopback(host):
        raise ValueError("shim: redirect target must be loopback, got %r" % host)
    _installed = (host, port)
    os.environ[ENV_REDIRECT] = "%s:%d" % (host, port)
    pp = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p != BOOT_DIR]
    os.environ["PYTHONPATH"] = os.pathsep.join([BOOT_DIR] + pp)
    for k in [k for k in os.environ if k.lower() in ("http_proxy", "https_proxy", "all_proxy", "no_proxy")]:
        del os.environ[k]
    target = (host, port)

    real_gai = socket.getaddrinfo
    real_cc = socket.create_connection
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_sendto = socket.socket.sendto
    real_sendmsg = getattr(socket.socket, "sendmsg", None)

    def getaddrinfo(h, p, family=0, type=0, proto=0, flags=0):
        return real_gai(host, port, socket.AF_INET, type, proto, flags)

    def create_connection(address, *a, **kw):
        kw.pop("all_errors", None)  # 3.11+ only; harmless to drop
        return real_cc(target, *a, **kw)

    def _rewrite(sock, addr):
        fam = getattr(sock, "family", None)
        if fam == socket.AF_INET:
            out = target
        elif fam == socket.AF_INET6:
            out = ("::ffff:" + host, port, 0, 0)
        else:
            return addr  # AF_UNIX etc.: local by nature
        if not _is_loopback(out[0]):
            raise PermissionError("shim: refusing non-loopback destination %r" % (addr,))
        return out

    def connect(self, addr):
        return real_connect(self, _rewrite(self, addr))

    def connect_ex(self, addr):
        return real_connect_ex(self, _rewrite(self, addr))

    def _dest_ok(self, addr):
        if getattr(self, "family", None) in (socket.AF_INET, socket.AF_INET6):
            if not (isinstance(addr, tuple) and addr and _is_loopback(addr[0])):
                raise PermissionError("shim: refusing datagram to %r" % (addr,))

    def sendto(self, data, *args):
        _dest_ok(self, args[-1])
        return real_sendto(self, data, *args)

    socket.getaddrinfo = getaddrinfo
    socket.create_connection = create_connection
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.socket.sendto = sendto
    if real_sendmsg is not None:
        def sendmsg(self, buffers, ancdata=(), flags=0, address=None):
            if address is not None:
                _dest_ok(self, address)
                return real_sendmsg(self, buffers, ancdata, flags, address)
            return real_sendmsg(self, buffers, ancdata, flags)
        socket.socket.sendmsg = sendmsg

    # ---- https -> plain http on loopback
    try:
        import ssl

        def wrap_socket(self, sock, *a, **kw):
            return sock
        ssl.SSLContext.wrap_socket = wrap_socket
    except ImportError:
        pass
    try:
        import http.client as hc
        if hasattr(hc, "HTTPSConnection"):
            hc.HTTPSConnection.connect = hc.HTTPConnection.connect
    except ImportError:
        pass

    _patch_spawning()


def _child_env(env):
    env = dict(os.environ if env is None else env)
    env[ENV_REDIRECT] = os.environ[ENV_REDIRECT]
    pp = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != BOOT_DIR]
    env["PYTHONPATH"] = os.pathsep.join([BOOT_DIR] + pp)
    for k in [k for k in env if k.lower() in ("http_proxy", "https_proxy", "all_proxy")]:
        del env[k]
    return env


def _child_argv(argv):
    """argv[0] must be a Python interpreter; drop flags that would skip sitecustomize."""
    argv = list(argv)
    if not argv or not _is_python(argv[0]):
        raise PermissionError("shim: rival may only start Python, not %r" % (argv[:1],))
    out = [argv[0]]
    i = 1
    while i < len(argv):
        a = str(argv[i])
        if a in ("-c", "-m") or not a.startswith("-") or a == "-":
            out.extend(argv[i:])
            break
        if not a.startswith("--") and len(a) > 1:
            a = "-" + "".join(ch for ch in a[1:] if ch not in "IESP")
            if a != "-":
                out.append(a)
        else:
            out.append(argv[i])
        i += 1
    return out


def _patch_spawning():
    import subprocess

    real_init = subprocess.Popen.__init__

    def popen_init(self, args, *a, **kw):
        if kw.get("shell"):
            raise PermissionError("shim: rival may not start a shell")
        exe = kw.get("executable")
        if isinstance(args, (str, bytes, os.PathLike)):
            args = [args]
        args = _child_argv(args)
        if exe is not None and not _is_python(exe):
            raise PermissionError("shim: rival may only start Python")
        # env is positional arg 9 of Popen; only handle the keyword form, refuse the rare positional one
        if len(a) >= 8 or (len(a) >= 2 and a[1] is not None and not _is_python(a[1])):
            raise PermissionError("shim: positional shell/env/executable to Popen is not supported")
        kw["env"] = _child_env(kw.get("env"))
        return real_init(self, args, *a, **kw)
    subprocess.Popen.__init__ = popen_init

    def _deny(*a, **kw):
        raise PermissionError("shim: rival may not start this kind of process")

    def system(cmd):
        return 127 << 8  # like a missing binary; never runs a shell
    os.system = system

    real_execve = os.execve

    def make_exec(name):
        def f(path, *rest):
            if name in ("execv", "execvp"):
                args, env = rest[0], None
            elif name in ("execve", "execvpe"):
                args, env = rest[0], rest[1]
            elif name in ("execl", "execlp"):
                args, env = rest, None
            else:  # execle, execlpe
                args, env = rest[:-1], rest[-1]
            if not _is_python(path):
                raise PermissionError("shim: rival may only exec Python, not %r" % (path,))
            exe = path if os.path.isabs(str(path)) or os.sep in str(path) else sys.executable
            return real_execve(exe, _child_argv([exe] + list(args)[1:]), _child_env(env))
        return f
    for name in ("execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe"):
        if hasattr(os, name):
            setattr(os, name, make_exec(name))
    for name in ("spawnv", "spawnve", "spawnvp", "spawnvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe",
                 "posix_spawn", "posix_spawnp", "forkpty"):
        if hasattr(os, name):
            setattr(os, name, _deny)
    # subprocess may use os.posix_spawn internally; we replaced it, so make it use fork_exec.
    try:
        subprocess._USE_POSIX_SPAWN = False
    except Exception:
        pass
    # _posixsubprocess.fork_exec is what multiprocessing (spawn/forkserver) and anyone bypassing
    # Popen call. subprocess >= 3.12 holds its own reference (already checked by Popen above).
    try:
        import _posixsubprocess as _ps
        real_fe = _ps.fork_exec

        def fork_exec(args, executable_list, *rest):
            if args is not None:
                dec = [os.fsdecode(x) for x in args]
                args = [os.fsencode(x) for x in _child_argv(dec)]
            for e in executable_list or ():
                if not _is_python(os.fsdecode(e)):
                    raise PermissionError("shim: rival may only start Python")
            rest = list(rest)
            if len(rest) >= 4 and rest[3] is not None:     # env_list: [b"K=V", ...]
                env = {}
                for kv in rest[3]:
                    k, _, v = os.fsdecode(kv).partition("=")
                    env[k] = v
                rest[3] = [os.fsencode("%s=%s" % kv) for kv in _child_env(env).items()]
            return real_fe(args, executable_list, *rest)
        _ps.fork_exec = fork_exec
    except Exception:
        pass
    # ctypes can call libc connect() directly, past every patch above. No rival bot path uses it.
    sys.modules["ctypes"] = None
    sys.modules["_ctypes"] = None


def _lo_up():
    """Bring up `lo` in this (fresh, empty) network namespace. Needs CAP_NET_ADMIN there (unshare -rn)."""
    import fcntl
    import struct
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        ifr = struct.pack("16sH22x", b"lo", 0)
        flags = struct.unpack("16sH22x", fcntl.ioctl(s.fileno(), 0x8913, ifr))[1]   # SIOCGIFFLAGS
        fcntl.ioctl(s.fileno(), 0x8914, struct.pack("16sH22x", b"lo", flags | 1))   # SIOCSIFFLAGS |= IFF_UP
    finally:
        s.close()


def start_forwarder(port, unix_path):
    """Inside a private network namespace (no route anywhere): listen on 127.0.0.1:port and relay each
    connection to the fake server's Unix socket, which is the only way out. Daemon threads."""
    import threading
    _lo_up()
    ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("127.0.0.1", int(port)))
    ls.listen(64)

    def pump(a, b):
        try:
            while True:
                d = a.recv(65536)
                if not d:
                    break
                b.sendall(d)
        except OSError:
            pass
        for x in (b, a):
            try:
                x.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def serve():
        while True:
            try:
                c, _ = ls.accept()
            except OSError:
                return
            try:
                u = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                u.connect(unix_path)
            except OSError:
                c.close()
                continue
            threading.Thread(target=pump, args=(c, u), daemon=True).start()
            threading.Thread(target=pump, args=(u, c), daemon=True).start()
    threading.Thread(target=serve, daemon=True).start()


def install_from_env():
    v = os.environ.get(ENV_REDIRECT)
    if not v:
        return False
    h, _, p = v.rpartition(":")
    install(h or "127.0.0.1", int(p))
    return True


def lifeline():
    """Our stdin is a pipe our parent (RivalHost) holds open: at EOF the parent is gone, so kill this whole
    process group (the rival and any children it started). The rival itself gets /dev/null as stdin."""
    import signal
    import threading
    fd = os.dup(0)
    nul = os.open(os.devnull, os.O_RDONLY)
    os.dup2(nul, 0)
    os.close(nul)

    def watch():
        try:
            while os.read(fd, 4096):
                pass
        except OSError:
            pass
        try:
            os.killpg(os.getpgid(0), signal.SIGKILL)
        except OSError:
            os._exit(1)
    threading.Thread(target=watch, daemon=True).start()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        sys.stderr.write(__doc__.split("\n\n")[1] + "\n")
        return 2
    i = argv.index("--")
    opts, rest = argv[:i], argv[i + 1:]
    redirect = None
    chdir = None
    via = None
    j = 0
    while j < len(opts):
        if opts[j] == "--redirect" and j + 1 < len(opts):
            redirect = opts[j + 1]
            j += 2
        elif opts[j] == "--via-unix" and j + 1 < len(opts):
            via = opts[j + 1]
            j += 2
        elif opts[j] == "--lifeline":
            lifeline()
            j += 1
        elif opts[j] == "--chdir" and j + 1 < len(opts):
            chdir = opts[j + 1]
            j += 2
        else:
            sys.stderr.write("shim: unknown option %r\n" % opts[j])
            return 2
    if not redirect or not rest:
        sys.stderr.write("shim: need --redirect HOST:PORT -- entry.py [args]\n")
        return 2
    h, _, p = redirect.rpartition(":")
    if via:
        start_forwarder(int(p), via)    # before install(): our own sockets must not be redirected
    install(h, int(p))
    if rest[0] == "--selftest":         # used by RivalHost to probe the namespace path
        import urllib.request
        with urllib.request.urlopen("https://selftest.invalid/api/v1/_ping", timeout=5) as r:
            sys.stdout.write("SELFTEST %s\n" % r.status)
        return 0
    entry = os.path.abspath(rest[0])
    os.chdir(chdir or os.path.dirname(entry))
    sys.argv = [rest[0]] + rest[1:]
    sys.path[:] = [os.path.dirname(entry)] + [x for x in sys.path if x and x != os.path.dirname(BOOT_DIR)]
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    sys.dont_write_bytecode = True
    import runpy
    runpy.run_path(entry, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
