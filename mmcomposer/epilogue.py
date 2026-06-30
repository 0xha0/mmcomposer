"""Elementwise epilogue description language (EDL) -- phase 1.

A tiny, pure, control-flow-free expression DSL the user writes as an ordinary
Python function (lambda or def) that takes ONE value and returns ONE value:

    from mmcomposer.epilogue import sigmoid
    silu = lambda x: x * sigmoid(x)

The function is *traced* once (called with a symbolic ``x``), building an
expression DAG, then *lowered* to a CUDA fp32 expression.  mmcomposer splices
that expression into the kernel epilogue so it runs per output element, in fp32,
right after the tensor-memory load and before the bf16 stage to SMEM/GMEM.

See ``mmcomposer/EPILOGUE.md`` for the formal language definition (semantics,
contract, grammar, builtins).

Two-tier builtins:
  * primitives  -- lower 1:1 to a CUDA intrinsic: exp, tanh, sqrt, log,
    abs (via ``abs(x)``), maximum, minimum.
  * composites  -- built *in the DSL* from primitives, no backend special-case:
    sigmoid(x) = 1/(1+exp(-x)), relu(x) = maximum(x, 0).

Public API:
    to_cuda(fn) -> str     # the CUDA fp32 expression in terms of `x`
    digest(fn)  -> str     # short stable hash (cache key / cubin tag)
    + the builtins: exp, tanh, sqrt, log, maximum, minimum, sigmoid, relu
"""
from __future__ import annotations

import hashlib
import inspect

# primitive op name -> CUDA intrinsic (1 arg unless noted)
_INTRINSIC = {
    "exp": "__expf",
    "tanh": "tanhf",
    "sqrt": "sqrtf",
    "log": "__logf",
    "abs": "fabsf",
    "maximum": "fmaxf",   # 2-arg
    "minimum": "fminf",   # 2-arg
}


class Expr:
    """A node in the epilogue expression DAG (a symbolic fp32 scalar)."""

    __slots__ = ("op", "args")

    def __init__(self, op: str, *args):
        self.op = op           # 'in'(idx) | 'const' | 'neg' | 'add'|'sub'|'mul'|'div'|'pow' | <primitive>
        self.args = args

    # -- input variables --
    # input 0 is the matmul accumulator ("x"); inputs 1.. are extra same-shape
    # operands ("c0", "c1", ...), e.g. lambda x, c: x * c  (phase 2).
    @staticmethod
    def input(i: int = 0) -> "Expr":
        return Expr("in", i)

    @staticmethod
    def inputs(n: int) -> tuple:
        return tuple(Expr("in", i) for i in range(n))

    # -- operator overloads (build the DAG) --
    def __add__(self, o):  return Expr("add", self, _wrap(o))
    def __radd__(self, o): return Expr("add", _wrap(o), self)
    def __sub__(self, o):  return Expr("sub", self, _wrap(o))
    def __rsub__(self, o): return Expr("sub", _wrap(o), self)
    def __mul__(self, o):  return Expr("mul", self, _wrap(o))
    def __rmul__(self, o): return Expr("mul", _wrap(o), self)
    def __truediv__(self, o):  return Expr("div", self, _wrap(o))
    def __rtruediv__(self, o): return Expr("div", _wrap(o), self)
    def __neg__(self):     return Expr("neg", self)
    def __pos__(self):     return self
    def __abs__(self):     return Expr("abs", self)

    def __pow__(self, e):
        if not isinstance(e, (int, float)) or isinstance(e, bool):
            raise TypeError("epilogue: exponent must be a numeric constant (no control flow)")
        return Expr("pow", self, e)

    # block accidental control flow / comparisons (phase 1 is straight-line math)
    def __bool__(self):
        raise TypeError(
            "epilogue functions are straight-line math: no if/and/or/comparisons "
            "on the value (use maximum/minimum instead of branching)")


def _wrap(v):
    if isinstance(v, Expr):
        return v
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"epilogue: operands must be Expr or numbers, got {type(v).__name__}")
    return Expr("const", float(v))


# -- builtins: primitives (1:1) --
def exp(x):  return Expr("exp", _wrap(x))
def tanh(x): return Expr("tanh", _wrap(x))
def sqrt(x): return Expr("sqrt", _wrap(x))
def log(x):  return Expr("log", _wrap(x))
def maximum(a, b): return Expr("maximum", _wrap(a), _wrap(b))
def minimum(a, b): return Expr("minimum", _wrap(a), _wrap(b))


# -- builtins: composites (defined over primitives, in the DSL itself) --
def sigmoid(x):
    x = _wrap(x)
    return 1.0 / (1.0 + exp(-x))


def relu(x):
    return maximum(_wrap(x), 0.0)


__all__ = ["Expr", "to_cuda", "to_torch", "digest", "arity", "n_inputs",
           "exp", "tanh", "sqrt", "log", "maximum", "minimum", "sigmoid", "relu"]


# ---- lowering: Expr DAG -> CUDA fp32 expression string --------------------
def _fmt_const(v: float) -> str:
    s = repr(float(v))
    return s if ("." in s or "e" in s or "E" in s or "inf" in s or "nan" in s) else s + ".0"


def _lower(e: "Expr") -> str:
    op = e.op
    if op == "in":
        i = e.args[0]
        return "x" if i == 0 else f"c{i - 1}"      # 0 -> accumulator; 1.. -> extra inputs
    if op == "const":
        return _fmt_const(e.args[0]) + "f"
    if op == "neg":
        return f"(-{_lower(e.args[0])})"
    if op in ("add", "sub", "mul"):
        sym = {"add": "+", "sub": "-", "mul": "*"}[op]
        return f"({_lower(e.args[0])} {sym} {_lower(e.args[1])})"
    if op == "div":
        # fast approximate division (rcp.approx, ~2 ULP) -- far below the bf16
        # output precision, and much cheaper than IEEE division.  This is what
        # makes sigmoid (1/(1+exp(-x))) fast, matching the hand-tuned kernels.
        return f"__fdividef({_lower(e.args[0])}, {_lower(e.args[1])})"
    if op == "pow":
        base, n = e.args
        # small non-negative integer power -> repeated multiply (avoids slow powf)
        if isinstance(n, int) and 0 <= n <= 8:
            if n == 0:
                return "1.0f"
            b = _lower(base)
            return "(" + " * ".join([b] * n) + ")"
        return f"powf({_lower(base)}, {_fmt_const(n)}f)"
    if op in _INTRINSIC:
        fn = _INTRINSIC[op]
        return f"{fn}({', '.join(_lower(a) for a in e.args)})"
    raise ValueError(f"epilogue: cannot lower op {op!r}")


def arity(fn) -> int:
    """Number of positional args of an epilogue callable (1 = accumulator only;
    n = accumulator + (n-1) extra same-shape inputs)."""
    if not callable(fn):
        raise TypeError("epilogue must be a callable (lambda or def)")
    params = list(inspect.signature(fn).parameters.values())
    for p in params:
        if p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            raise TypeError("epilogue takes only positional args (one per input); "
                            "no *args/**kwargs/keyword-only")
    if not params:
        raise TypeError("epilogue must take at least one arg (the accumulator x)")
    return len(params)


def n_inputs(fn) -> int:
    """Number of *extra* inputs the epilogue expects (arity - 1)."""
    return arity(fn) - 1


def to_cuda(fn) -> str:
    """Trace `fn` (n Exprs in -> one Expr/number out) and return its CUDA fp32
    expression in terms of ``x`` (input 0) and ``c0``, ``c1``, ... (extra inputs)."""
    y = fn(*Expr.inputs(arity(fn)))
    if isinstance(y, tuple):
        raise TypeError("epilogue must return a single value, not a tuple")
    if not isinstance(y, Expr):
        y = _wrap(y)           # a constant epilogue, e.g. lambda x: 0.0
    return _lower(y)


def digest(fn) -> str:
    """Short stable hash of the epilogue (for cache keys / cubin tags)."""
    return hashlib.sha1(to_cuda(fn).encode()).hexdigest()[:10]


# ---- second backend: lower the same DAG to torch (the verify reference) ----
def _lower_torch(e: "Expr", ins, torch):
    op = e.op
    if op == "in":
        return ins[e.args[0]]                      # input tensor by index
    if op == "const":
        return float(e.args[0])
    if op == "neg":
        return -_lower_torch(e.args[0], ins, torch)
    if op in ("add", "sub", "mul", "div"):
        a = _lower_torch(e.args[0], ins, torch)
        b = _lower_torch(e.args[1], ins, torch)
        return {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[op]
    if op == "pow":
        return _lower_torch(e.args[0], ins, torch) ** e.args[1]
    if op in ("abs", "exp", "tanh", "sqrt", "log"):
        fn = {"abs": torch.abs, "exp": torch.exp, "tanh": torch.tanh,
              "sqrt": torch.sqrt, "log": torch.log}[op]
        return fn(_lower_torch(e.args[0], ins, torch))
    if op in ("maximum", "minimum"):
        a = _lower_torch(e.args[0], ins, torch)
        b = _lower_torch(e.args[1], ins, torch)
        ref = ins[0]                               # tensor to match dtype/device
        if not torch.is_tensor(a):
            a = torch.as_tensor(a, dtype=ref.dtype, device=ref.device)
        if not torch.is_tensor(b):
            b = torch.as_tensor(b, dtype=ref.dtype, device=ref.device)
        return (torch.maximum if op == "maximum" else torch.minimum)(a, b)
    raise ValueError(f"epilogue: cannot lower op {op!r} to torch")


def to_torch(fn):
    """Trace `fn` and return a callable applying the same elementwise op to torch
    tensors -- the reference used to verify a fused epilogue kernel, mirroring
    `to_cuda` (in higher precision; uses torch's exact exp/div, not fast intrinsics).
    The returned callable takes the accumulator tensor then the extra inputs:
    ``ref_fn(a @ b, c0, c1, ...)``."""
    import torch
    n = arity(fn)
    y = fn(*Expr.inputs(n))
    if isinstance(y, tuple):
        raise TypeError("epilogue must return a single value, not a tuple")
    if not isinstance(y, Expr):
        y = _wrap(y)
    return lambda *ins: _lower_torch(y, ins, torch)
