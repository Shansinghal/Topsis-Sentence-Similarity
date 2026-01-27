import sys
import pandas as pd
import numpy as np


def error(message):
    print(f"Error: {message}")
    sys.exit(1)


def topsis(input_file, weights, impacts, output_file):
    # Read input file
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        error("Input file not found")

    # Check minimum columns
    if df.shape[1] < 3:
        error("Input file must contain at least three columns")

    # Separate numeric data
    try:
        data = df.iloc[:, 1:].astype(float)
    except ValueError:
        error("All columns except the first must contain numeric values")

    # Parse weights and impacts
    weights = weights.split(",")
    impacts = impacts.split(",")

    if len(weights) != len(impacts) or len(weights) != data.shape[1]:
        error("Number of weights, impacts, and criteria must be equal")

    try:
        weights = np.array(weights, dtype=float)
    except ValueError:
        error("Weights must be numeric")

    for impact in impacts:
        if impact not in ['+', '-']:
            error("Impacts must be either '+' or '-'")

    # Step 1: Normalize the decision matrix
    norm = np.sqrt((data ** 2).sum())
    normalized = data / norm

    # Step 2: Apply weights
    weighted = normalized * weights

    # Step 3: Determine ideal best and worst
    ideal_best = []
    ideal_worst = []

    for i in range(len(impacts)):
        if impacts[i] == '+':
            ideal_best.append(weighted.iloc[:, i].max())
            ideal_worst.append(weighted.iloc[:, i].min())
        else:
            ideal_best.append(weighted.iloc[:, i].min())
            ideal_worst.append(weighted.iloc[:, i].max())

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    # Step 4: Calculate separation measures
    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    # Step 5: Calculate TOPSIS score
    score = dist_worst / (dist_best + dist_worst)

    # Step 6: Rank alternatives
    df["Topsis Score"] = score
    df["Rank"] = df["Topsis Score"].rank(ascending=False, method="dense").astype(int)

    # Save output
    df.to_csv(output_file, index=False)
    print("TOPSIS analysis completed successfully.")


def main():
    if len(sys.argv) != 5:
        print("Usage:")
        print("python topsis.py <input_file> <weights> <impacts> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    weights = sys.argv[2]
    impacts = sys.argv[3]
    output_file = sys.argv[4]

    topsis(input_file, weights, impacts, output_file)


if __name__ == "__main__":
    main()
