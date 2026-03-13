"""Benchmark: grep/read vs v-code for common code exploration tasks.

Measures wall-clock time and output quality for typical Claude Code agent tasks.
"""
import subprocess
import time
import os

PROJECT = r"C:\Users\sangm\Desktop\project\hnsw-model2vec"
DB = ".v-hnsw-code.db"

def run(cmd, cwd=PROJECT, shell=True):
    start = time.perf_counter()
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, shell=shell, timeout=30)
    elapsed = time.perf_counter() - start
    output = r.stdout + r.stderr
    lines = len(output.strip().split("\n")) if output.strip() else 0
    return elapsed, lines, output

def task(name, grep_cmds, vcode_cmd):
    """Run a task with grep/read approach vs v-code approach."""
    print(f"\n{'─'*60}")
    print(f"  Task: {name}")
    print(f"{'─'*60}")

    # grep/read approach (multiple commands)
    grep_total = 0
    grep_lines = 0
    grep_output = ""
    for cmd in grep_cmds:
        t, l, o = run(cmd)
        grep_total += t
        grep_lines += l
        grep_output += o

    # v-code approach (single command)
    vc_time, vc_lines, vc_output = run(vcode_cmd)

    print(f"  grep/read: {grep_total*1000:.0f}ms, {grep_lines} output lines, {len(grep_cmds)} commands")
    print(f"  v-code:    {vc_time*1000:.0f}ms, {vc_lines} output lines, 1 command")
    ratio = grep_total / max(vc_time, 0.001)
    if ratio > 1:
        print(f"  → v-code {ratio:.1f}x faster")
    else:
        print(f"  → grep {1/ratio:.1f}x faster")

    return {
        "name": name,
        "grep_ms": grep_total * 1000,
        "grep_lines": grep_lines,
        "grep_cmds": len(grep_cmds),
        "vcode_ms": vc_time * 1000,
        "vcode_lines": vc_lines,
    }

results = []

# Task 1: Find symbol definition
results.append(task(
    "심볼 정의 찾기 (StorageEngine::open)",
    [
        'rg -n "pub fn open" --type rust',
        'rg -n "fn open" --type rust -g "engine*"',
    ],
    f"v-code def {DB} StorageEngine::open",
))

# Task 2: Find all callers of a function
results.append(task(
    "함수 호출자 찾기 (StorageEngine::open callers)",
    [
        'rg -n "StorageEngine::open" --type rust',
        'rg -n "engine\\.open\\(" --type rust',
    ],
    f"v-code impact {DB} StorageEngine::open --depth 1",
))

# Task 3: Understand call chain (who calls what)
results.append(task(
    "호출 체인 추적 (insert → insert_core → ?)",
    [
        'rg -n "insert_core" --type rust -g "insert*"',
        'rg -n "fn insert_core" --type rust',
        'rg -n "insert_core\\(" --type rust',
    ],
    f"v-code gather {DB} insert_core --depth 2",
))

# Task 4: Find all symbols in a module
results.append(task(
    "모듈 심볼 목록 (delta.rs)",
    [
        'rg -n "pub fn |pub struct |pub enum |pub trait " --type rust -g "delta*"',
    ],
    f"v-code symbols {DB} --name delta",
))

# Task 5: File dependency graph
results.append(task(
    "파일 의존성 분석 (loader.rs)",
    [
        'rg -n "use |mod " --type rust -g "loader*"',
        'rg -n "loader" --type rust',
    ],
    f"v-code deps {DB} loader",
))

# Task 6: Find path between two symbols
results.append(task(
    "심볼 간 경로 (StorageEngine::open → load_chunks)",
    [
        'rg -n "StorageEngine::open" --type rust',
        'rg -n "load_chunks" --type rust',
        'rg -n "load_chunks_from_db" --type rust',
    ],
    f"v-code trace {DB} StorageEngine::open load_chunks",
))

# Task 7: Code search (semantic)
results.append(task(
    "코드 검색 (vector distance calculation)",
    [
        'rg -n "distance" --type rust -g "*.rs" | head -30',
    ],
    f"v-code find {DB} \"vector distance calculation\" -k 5",
))

# Task 8: Duplicate detection
results.append(task(
    "중복 코드 감지 (전체)",
    [
        # grep approach: impossible in practice
        'echo "grep cannot detect code clones"',
    ],
    f"v-code dupes {DB} --threshold 0.6 -k 10",
))

# Summary
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"{'Task':<45} {'grep(ms)':>8} {'v-code(ms)':>10} {'Winner':>8}")
print(f"{'─'*45} {'─'*8} {'─'*10} {'─'*8}")

grep_wins = 0
vcode_wins = 0
for r in results:
    winner = "v-code" if r["vcode_ms"] < r["grep_ms"] else "grep"
    if winner == "v-code":
        vcode_wins += 1
    else:
        grep_wins += 1
    print(f"{r['name']:<45} {r['grep_ms']:>7.0f} {r['vcode_ms']:>9.0f}  {winner:>7}")

print(f"\n  grep wins: {grep_wins}, v-code wins: {vcode_wins}")
print(f"\n  Key advantages of v-code:")
print(f"  - Type-resolved call graph (MIR) — grep cannot distinguish Vec::push from DeltaNeighbors::push")
print(f"  - Impact analysis, gather, trace — impossible with grep alone")
print(f"  - Single command vs multi-step grep chains")
print(f"  - Clone detection — grep cannot do this")
