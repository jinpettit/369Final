import matplotlib.pyplot as plt
import numpy as np

def merge_dicts(dict1, dict2) -> dict:
    return(dict1.update(dict2))

def read_data(filename: str) -> dict[str, (float, float)]:
    table = {}      # {country, (expenditure, completion_rate)}
    with open(filename, "r") as f:
        for line in f:
            tokens = line.split(",")
            country = tokens[0]
            expenditure = float(tokens[1])
            completion = float(tokens[2])
            table[country] = (expenditure, completion)
    return table

def main():
    lm_slope = 0.5318476732701783
    lm_intercept = 75.88447724457082

    table = read_data("joined_data_output/part-00000")
    merge_dicts(table, read_data("joined_data_output/part-00001"))

    x_values = [value[0] for value in table.values()]
    y_values = [value[1] for value in table.values()]

    x_line = np.linspace(min(x_values), max(x_values), 100)
    y_line = lm_slope * x_line + lm_intercept

    plt.scatter(x_values, y_values)
    plt.plot(x_line, y_line, color="red")
    plt.title("Primary School Completion Rate vs. Government Expenditure")
    plt.xlabel("Government Expenditure on Children's Education (% of GDP per Capita)")
    plt.ylabel("Primary School Completion Rate")
    plt.show()


if __name__ == "__main__":
    main()