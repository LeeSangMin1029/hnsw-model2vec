"""Extract references from a SCIP index for comparison with MIR call graph.

Usage: python scripts/scip_refs.py <symbol_suffix> [--callers|--callees]
"""
import sys
sys.path.insert(0, r"C:\Users\sangm\AppData\Local\Temp")
import scip_pb2  # noqa: E402

def load_index(path="index.scip"):
    idx = scip_pb2.Index()
    with open(path, "rb") as f:
        idx.ParseFromString(f.read())
    return idx

def parse_symbol_name(symbol: str) -> str:
    """Extract human-readable name from SCIP symbol string."""
    # SCIP symbol format: "rust-analyzer cargo <pkg> <ver> <path>"
    parts = symbol.split(" ")
    if len(parts) >= 5:
        return " ".join(parts[4:]).rstrip("/").rstrip(".")
    return symbol

def find_refs(index, target_suffix):
    """Find all references to symbols matching the suffix."""
    # Build symbol → definition location map
    sym_defs = {}  # symbol -> (file, line)
    sym_refs = {}  # symbol -> [(file, line)]

    # Collect all symbols and their occurrences
    for doc in index.documents:
        filepath = doc.relative_path
        for occ in doc.occurrences:
            symbol = occ.symbol
            if not symbol or symbol.startswith("local "):
                continue

            # Role: 1 = Definition, 0 = Reference
            is_def = (occ.symbol_roles & 1) != 0
            line = occ.range[0] if occ.range else 0

            if is_def:
                sym_defs[symbol] = (filepath, line + 1)
            else:
                sym_refs.setdefault(symbol, []).append((filepath, line + 1))

    # Find symbols matching the target
    target_lower = target_suffix.lower()
    matching_symbols = []
    for sym in sym_defs:
        readable = parse_symbol_name(sym)
        if target_lower in readable.lower():
            matching_symbols.append((sym, readable, sym_defs[sym]))

    return matching_symbols, sym_refs

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/scip_refs.py <symbol_name>")
        sys.exit(1)

    target = sys.argv[1]
    print(f"Loading SCIP index...")
    index = load_index()
    print(f"  {len(index.documents)} documents")

    matching, all_refs = find_refs(index, target)

    if not matching:
        print(f"No symbols matching '{target}' found.")
        return

    print(f"\nSymbols matching '{target}':")
    for sym, readable, (filepath, line) in matching:
        print(f"\n  [{readable}]")
        print(f"    def: {filepath}:{line}")
        refs = all_refs.get(sym, [])
        # Group refs by file
        by_file = {}
        for rfile, rline in refs:
            by_file.setdefault(rfile, []).append(rline)

        if refs:
            print(f"    {len(refs)} references:")
            for rfile in sorted(by_file):
                lines = sorted(by_file[rfile])
                print(f"      {rfile}: lines {lines}")
        else:
            print(f"    0 references")

if __name__ == "__main__":
    main()
