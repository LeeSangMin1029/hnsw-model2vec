"""Compare SCIP references vs MIR call graph for specific symbols."""
import sys
import subprocess
import json
sys.path.insert(0, r"C:\Users\sangm\AppData\Local\Temp")
import scip_pb2

def load_scip():
    idx = scip_pb2.Index()
    with open("index.scip", "rb") as f:
        idx.ParseFromString(f.read())
    return idx

def scip_callers(index, target_suffix):
    """Find all reference locations for a symbol matching target_suffix."""
    target_sym = None
    for doc in index.documents:
        for occ in doc.occurrences:
            if not occ.symbol or "local " in occ.symbol:
                continue
            if (occ.symbol_roles & 1) and target_suffix.lower() in occ.symbol.lower():
                target_sym = occ.symbol
                break
        if target_sym:
            break

    if not target_sym:
        return [], target_sym

    refs = []
    for doc in index.documents:
        for occ in doc.occurrences:
            if occ.symbol == target_sym and not (occ.symbol_roles & 1):
                refs.append((doc.relative_path, occ.range[0] + 1))
    return refs, target_sym

def mir_callers(symbol):
    """Get callers from v-code impact (depth=1, non-test only)."""
    result = subprocess.run(
        ["./target/debug/v-code", "impact", ".v-hnsw-code.db", symbol, "--depth", "1", "--format", "json"],
        capture_output=True, text=True
    )
    try:
        data = json.loads(result.stdout)
    except:
        return []

    callers = []
    for item in data.get("results", []):
        if item.get("d", 0) == 1:  # depth 1 = direct callers
            callers.append((item["f"], item["l"], item["n"], item.get("t", False)))
    return callers

def compare_symbol(index, scip_suffix, mir_name):
    print(f"\n{'='*60}")
    print(f"  {mir_name}")
    print(f"{'='*60}")

    scip_refs, sym = scip_callers(index, scip_suffix)
    mir_refs = mir_callers(mir_name)

    # Group SCIP refs by file
    scip_by_file = {}
    for fpath, line in scip_refs:
        fpath_norm = fpath.replace("\\", "/")
        scip_by_file.setdefault(fpath_norm, []).append(line)

    # Separate SCIP into prod vs test
    def is_test_or_bench(f):
        return "/tests/" in f or "\\tests\\" in f or "/benches/" in f or "\\benches\\" in f
    scip_prod_files = {f for f in scip_by_file if not is_test_or_bench(f)}
    scip_test_files = {f for f in scip_by_file if is_test_or_bench(f)}

    # MIR files
    mir_prod = [(r[0], r[1], r[2]) for r in mir_refs if not r[3]]
    mir_test = [(r[0], r[1], r[2]) for r in mir_refs if r[3]]
    mir_prod_files = {r[0] for r in mir_prod}
    mir_test_files = {r[0] for r in mir_test}

    print(f"\nSCIP: {len(scip_prod_files)} prod files, {len(scip_test_files)} test files ({len(scip_refs)} total refs)")
    print(f"MIR:  {len(mir_prod_files)} prod files, {len(mir_test_files)} test files ({len(mir_refs)} total callers)")

    # File-level comparison (prod only)
    both = scip_prod_files & mir_prod_files
    only_scip = scip_prod_files - mir_prod_files
    only_mir = mir_prod_files - scip_prod_files

    if both:
        print(f"\n  Match ({len(both)}):")
        for f in sorted(both):
            scip_lines = sorted(scip_by_file.get(f, []))
            mir_fn = [r[2] for r in mir_prod if r[0] == f]
            print(f"    {f}  SCIP lines:{scip_lines}  MIR fn:{mir_fn}")

    if only_scip:
        print(f"\n  SCIP only ({len(only_scip)}):")
        for f in sorted(only_scip):
            lines = sorted(scip_by_file.get(f, []))
            print(f"    {f}: lines {lines}")

    if only_mir:
        print(f"\n  MIR only ({len(only_mir)}):")
        for f in sorted(only_mir):
            mir_fn = [r[2] for r in mir_prod if r[0] == f]
            print(f"    {f}: {mir_fn}")

    if not only_scip and not only_mir:
        print(f"\n  PERFECT MATCH (prod files)")

    return len(both), len(only_scip), len(only_mir)

def main():
    print("Loading SCIP index...")
    index = load_scip()

    symbols = [
        ("DeltaNeighbors]push", "DeltaNeighbors::push"),
        ("insert/insert_core", "insert_core"),
        ("L2Distance][DistanceMetric]distance()", "L2Distance::distance"),
        ("StorageEngine]open()", "StorageEngine::open"),
        ("CallGraph]build()", "CallGraph::build"),
        ("load_chunks_from_db", "load_chunks_from_db"),
        ("insert/insert()", "insert"),
        ("AutoDistance]distance()", "AutoDistance::distance"),
    ]

    total_both = total_scip = total_mir = 0
    for scip_s, mir_s in symbols:
        b, s, m = compare_symbol(index, scip_s, mir_s)
        total_both += b
        total_scip += s
        total_mir += m

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Matching files: {total_both}")
    print(f"  SCIP-only files: {total_scip}")
    print(f"  MIR-only files: {total_mir}")
    accuracy = total_both / max(1, total_both + total_scip + total_mir) * 100
    print(f"  Accuracy: {accuracy:.1f}%")

if __name__ == "__main__":
    main()
