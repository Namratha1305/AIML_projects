import sys

def parse_input(input_str):
    try:
        return [int(x.strip()) for x in input_str.split(',')]
    except ValueError:
        print("Invalid input. Please enter a comma-separated list of integers.")
        sys.exit(1)

def predict_next(seq):
    n = len(seq)
    if n < 2:
        print("Need at least two numbers to predict the next in sequence.")
        sys.exit(1)

    # Check for arithmetic progression
    diffs = [seq[i+1] - seq[i] for i in range(n-1)]
    if all(d == diffs[0] for d in diffs):
        return seq[-1] + diffs[0]

    # Check for geometric progression
    ratios = []
    for i in range(n-1):
        if seq[i] == 0:
            break
        ratios.append(seq[i+1] / seq[i])
    if len(ratios) == n-1 and all(r == ratios[0] for r in ratios):
        return int(seq[-1] * ratios[0])

    # Fallback: use the last difference
    return seq[-1] + (seq[-1] - seq[-2])

def main():
    input_str = input("Enter a sequence of integers (comma-separated): ")
    seq = parse_input(input_str)
    next_num = predict_next(seq)
    print(f"The next number in the sequence is: {next_num}")

if __name__ == "__main__":
    main() 