"""Predict entailment labels for a private CSV test set.

Usage:
    python3 run_private.py <notebook.ipynb> <private_test.csv> [submission.csv]

For each row:
  1. RuleTaker and causal rows: exact solvers on the `context_logic` /
     `statement_logic` columns (forward chaining; structural causal model).
  2. First-order logic rows, and anything that cannot be parsed: the notebook's
     ClassicalReasoningAgent on the natural-language `context` / `statement`.

Self-contained: needs only Python 3 and the notebook file.

Writes id,label to the output CSV and prints accuracy if a `label` column exists.
"""
import itertools
import re
import collections
import contextlib
import csv
import io
import json
import os
import sys
import time

# The notebook's agent iterates over sets, so results depend on Python's hash
# seed. Restart once with a fixed seed so every run gives the same predictions.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)


# =============================================================================
# 1. RuleTaker (closed world, negation as failure)
# =============================================================================
_VARS = {"someone", "something"}


def _parse_atom(s):
    toks = re.findall(r'"([^"]*)"', s)
    if len(toks) != 4 or toks[3] not in "+-":
        raise ValueError(f"bad atom: {s}")
    return (toks[0], toks[1], toks[2]), toks[3] == "+"


def _parse_ruletaker(context):
    facts, rules = set(), []
    for line in context.splitlines():
        line = line.strip()
        if not line:
            continue
        if "->" in line:
            lhs, rhs = line.split("->", 1)
            body = [_parse_atom(a) for a in re.findall(r'\((?:\s*"[^"]*")+\s*\)', lhs)]
            head = _parse_atom(rhs)
            rules.append((body, head))
        else:
            atom, pos = _parse_atom(line)
            if pos:
                facts.add(atom)
            # explicit negative facts are already implied by the closed world
    return facts, rules


def _unify(pattern, atom, env):
    env = dict(env)
    for p, a in zip(pattern, atom):
        if p in _VARS:
            if env.get(p, a) != a:
                return None
            env[p] = a
        elif p != a:
            return None
    return env


def _subst(pattern, env):
    return tuple(env.get(t, t) if t in _VARS else t for t in pattern)


def _key(atom):  # predicate identity used for stratification
    return (atom[1], "*" if atom[2] in _VARS else atom[2])


def _keys_match(a, b):
    return a[0] == b[0] and (a[1] == b[1] or "*" in (a[1], b[1]))


def _stratify(rules):
    strata = [0] * len(rules)
    for _ in range(len(rules) + 1):
        changed = False
        for i, (body, _h) in enumerate(rules):
            for atom, pos in body:
                for j, (_b, head) in enumerate(rules):
                    if _keys_match(_key(atom), _key(head[0])):
                        need = strata[j] + (0 if pos else 1)
                        if strata[i] < need:
                            strata[i], changed = need, True
        if not changed:
            break
    return strata


def _forward_chain(facts, rules):
    known = set(facts)
    entities = {a[0] for a in known} | {a[2] for a in known}
    for _b, (h, _p) in rules:
        entities |= {t for t in (h[0], h[2]) if t not in _VARS}
    for body, _h in rules:
        for (a, _p) in body:
            entities |= {t for t in (a[0], a[2]) if t not in _VARS}
    strata = _stratify(rules)
    for level in sorted(set(strata)):
        layer = [r for r, s in zip(rules, strata) if s == level]
        changed = True
        while changed:
            changed = False
            for body, (head, head_pos) in layer:
                pos_body = [a for a, p in body if p]
                neg_body = [a for a, p in body if not p]
                envs = [{}]
                for atom in pos_body:
                    envs = [e2 for e in envs for f in known
                            for e2 in [_unify(atom, f, e)] if e2 is not None]
                # bind variables that appear only in negated / head atoms
                free = {t for a in neg_body + [head] for t in (a[0], a[2]) if t in _VARS}
                full = []
                for e in envs:
                    missing = [v for v in free if v not in e]
                    for combo in itertools.product(entities, repeat=len(missing)):
                        full.append({**e, **dict(zip(missing, combo))})
                for e in full:
                    if all(_subst(a, e) not in known for a in neg_body):
                        new = _subst(head, e)
                        if head_pos and new not in known:
                            known.add(new)
                            changed = True
    return known


def solve_ruletaker(context, statement):
    facts, rules = _parse_ruletaker(context)
    query, positive = _parse_atom(statement)
    derived = query in _forward_chain(facts, rules)
    return derived if positive else not derived


# =============================================================================
# 2. Causal counterfactuals over Boolean structural equations
# =============================================================================
def _parse_causal(context):
    eqs, evidence = {}, {}
    for line in context.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("evidence:"):
            for k, v in re.findall(r"(\w+)\s*=\s*([01])", line):
                evidence[k] = v == "1"
            continue
        lhs, rhs = (s.strip() for s in line.split("=", 1))
        eqs[lhs] = rhs
    return eqs, evidence


def _eval_expr(expr, vals):
    py = re.sub(r"\bAND\b", " and ", expr)
    py = re.sub(r"\bOR\b", " or ", py)
    py = re.sub(r"\bNOT\b", " not ", py)
    py = re.sub(r"\b([A-Za-z_]\w*)\b",
                lambda m: m.group(1) if m.group(1) in ("and", "or", "not")
                else f"V[{m.group(1)!r}]", py)
    return bool(eval(py, {"__builtins__": {}}, {"V": vals}))


def _simulate(eqs, roots, interventions):
    vals = dict(roots)
    vals.update(interventions)
    pending = [v for v in eqs if v not in interventions]
    for _ in range(len(eqs) + 1):
        for v in list(pending):
            try:
                vals[v] = _eval_expr(eqs[v], vals)
                pending.remove(v)
            except KeyError:
                pass
    return vals


def solve_causal(context, statement):
    eqs, evidence = _parse_causal(context)
    m = re.match(r"\s*(\w+)_\{([^}]*)\}\s*=\s*([01])", statement)
    if not m:
        raise ValueError(f"bad causal query: {statement}")
    target, target_val = m.group(1), m.group(3) == "1"
    do = {k: v == "1" for k, v in re.findall(r"(\w+)\s*=\s*([01])", m.group(2))}
    names = set(eqs) | set(evidence) | set(do)
    for rhs in eqs.values():
        names |= set(re.findall(r"\b(?!AND\b|OR\b|NOT\b)[A-Za-z_]\w*", rhs))
    roots = sorted(n for n in names if n not in eqs)
    unknown = [r for r in roots if r not in evidence]
    answers = []
    for combo in itertools.product([False, True], repeat=len(unknown)):
        u = {r: evidence[r] for r in roots if r in evidence}
        u.update(zip(unknown, combo))
        factual = _simulate(eqs, u, {})
        if any(factual.get(k) != v for k, v in evidence.items()):
            continue  # abduction: keep only worlds consistent with the evidence
        answers.append(_simulate(eqs, u, do)[target] == target_val)
    return bool(answers) and all(answers)




def detect_format(context_logic):
    c = (context_logic or "").strip()
    if not c:
        return None
    if c.startswith('("') or c.startswith('(("'):
        return "ruletaker"
    if re.search(r"^Evidence:", c, flags=re.M) or re.search(r"^\w+ = \w+", c, flags=re.M):
        return "causal"
    return "fol"


def solve_row(context_logic, statement_logic):
    try:
        kind = detect_format(context_logic)
        if kind == "ruletaker":
            return solve_ruletaker(context_logic, statement_logic)
        if kind == "causal":
            return solve_causal(context_logic, statement_logic)
    except Exception:
        return None
    return None


# =============================================================================
# Runner
# =============================================================================
def _find(name, explicit):
    """Use the path given on the command line, else look in the usual places."""
    if explicit:
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (here, os.getcwd(), os.path.expanduser("~/Desktop/AI"),
                   os.path.expanduser("~/Desktop/Assignment"), os.path.expanduser("~/Downloads")):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return p
    sys.exit(f"Could not find {name}. Run: python3 {os.path.basename(__file__)} "
             f"<notebook.ipynb> <private_test.csv> [submission.csv]")


_args = sys.argv[1:]
nb_path = _find("reasoning_agent.ipynb", _args[0] if len(_args) > 0 else None)
csv_path = _find("private_test.csv", _args[1] if len(_args) > 1 else None)
out_path = _args[2] if len(_args) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(csv_path)), "submission.csv")
print(f"Notebook: {nb_path}\nTest CSV: {csv_path}")

_agent = None
_ns = {"__name__": "reasoning_agent"}


def notebook_agent():
    """Load the notebook's agent (code cells 0-4) only if it is needed."""
    global _agent
    if _agent is None:
        nb = json.load(open(nb_path, encoding="utf-8"))
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in code_cells[:5]:
                exec("".join(cell["source"]), _ns)
        _agent = _ns["ClassicalReasoningAgent"](search_strategy="astar")
    return _agent


rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
has_gold = bool(rows) and "label" in rows[0]

# Load the notebook's agent once, before timing, so inference time excludes setup.
t_load = time.perf_counter()
notebook_agent()
load_time = time.perf_counter() - t_load

out = []
stats = collections.defaultdict(lambda: [0, 0, 0.0])  # kind -> [correct, total, seconds]
total_time = 0.0
for i, r in enumerate(rows):
    t_row = time.perf_counter()
    kind = detect_format(r.get("context_logic", ""))
    pred = None
    if kind in ("ruletaker", "causal"):
        # Exact solvers on the structured logic columns.
        pred = solve_row(r.get("context_logic", ""), r.get("statement_logic", ""))
    # First-order logic rows (and anything unparsed) use the notebook's agent
    # on the natural-language `context` / `statement` below.
    if pred is None:
        kind = f"{kind or 'nl'}->notebook"
        try:
            pred = bool(notebook_agent().predict_record(
                {"context": r.get("context", ""), "statement": r.get("statement", "")}))
        except Exception as e:
            print(f"[warn] row {r.get('id', i)} failed: {e!r}; predicting False")
            pred = False
    row_time = time.perf_counter() - t_row
    total_time += row_time
    stats[kind][1] += 1
    stats[kind][2] += row_time
    out.append({"id": r.get("id", i), "label": "True" if pred else "False"})
    if has_gold:
        gold = str(r["label"]).strip().lower() in ("true", "entailment")
        stats[kind][0] += int(gold == pred)
        if gold != pred:
            print(f"[miss] {r.get('id', i)} ({kind}) gold={gold} pred={pred}")

with open(out_path, "w", newline="", encoding="utf-8") as fp:
    w = csv.DictWriter(fp, fieldnames=["id", "label"])
    w.writeheader()
    w.writerows(out)

n = len(rows)
print("=" * 72)
print(f"Wrote {n} predictions to {out_path}")
print(f"Agent load time       : {load_time:.4f}s (not counted below)")
print("-" * 72)
print(f"{'Question type':<20} {'Accuracy':>14} {'Time (s)':>10} {'ms/query':>10}")
for kind, (c, k, t) in sorted(stats.items()):
    acc = f"{c}/{k} ({100 * c / k:5.1f}%)" if has_gold else "-"
    print(f"{kind:<20} {acc:>14} {t:>10.4f} {1000 * t / k:>10.3f}")
print("-" * 72)
if has_gold and n:
    c = sum(v[0] for v in stats.values())
    print(f"Accuracy              : {c}/{n} ({100 * c / n:.2f}%)")
if n:
    print(f"Total inference time  : {total_time:.4f}s ({1000 * total_time / n:.3f} ms/query)")
print("=" * 72)
