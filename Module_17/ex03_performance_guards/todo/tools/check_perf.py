import json
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python check_perf.py <results.json> <min_throughput>")
        sys.exit(1)

    json_path = sys.argv[1]
    min_throughput = float(sys.argv[2])

    # TODO: Load JSON file
    # TODO: Iterate over benchmarks
    # TODO: Check 'items_per_second' vs min_throughput
    # TODO: Exit 1 if too slow
    
    print("TODO: Implement check")
    sys.exit(1) # Fail by default

if __name__ == "__main__":
    main()
