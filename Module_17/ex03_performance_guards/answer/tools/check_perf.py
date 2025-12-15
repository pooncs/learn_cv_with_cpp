import json
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python check_perf.py <results.json> <min_throughput>")
        sys.exit(1)

    json_path = sys.argv[1]
    min_throughput = float(sys.argv[2])

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        sys.exit(1)

    found = False
    for b in data['benchmarks']:
        if 'BM_ProcessImage' in b['name']:
            found = True
            throughput = b.get('items_per_second', 0)
            print(f"Benchmark: {b['name']}, Throughput: {throughput:.2f} items/s, Threshold: {min_throughput}")
            if throughput < min_throughput:
                print("FAIL: Performance regression detected!")
                sys.exit(1)
            else:
                print("PASS: Performance checks passed.")
    
    if not found:
        print("Error: Benchmark 'BM_ProcessImage' not found in results.")
        sys.exit(1)

if __name__ == "__main__":
    main()
