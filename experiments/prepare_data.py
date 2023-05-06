import os

# Parse csv file to tuple of (header, entries), does some sanity checks on the data. Each entry should be N features followed by 1 target
def parse_dataset(csv_path):
    # Return mapping function that checks size of each line and checks binary in size - 1 first columns
    def map_parse_line_for_size(size):
        def map_parse_line(line: str):
            segments = line.strip().split(",")
            if len(segments) != size:
                print(f"Invalid size of line {line}\n")
            for i in segments[0:size - 1]:
                if i != "0" and i != "1":
                    print(f"Non-binary feature in line {line}\n")
            return segments
            
        return map_parse_line

    with open(csv_path) as csv:
        header = csv.readline().strip().split(",")
        return (header, list(map(map_parse_line_for_size(len(header)), csv.readlines())))

def write_osrt_dataset(dataset, path):
    header, entries = dataset
    lines = [",".join(header) + "\n"]
    lines.extend([",".join(entry) + "\n" for entry in entries])
    with open(path, "w") as csv:
        csv.writelines(lines)

def write_streed_dataset(dataset, path):
    header, entries = dataset
    N = len(header)
    # Transpose target and features
    lines = ([" ".join([entry[N - 1]] + entry[0:N - 1]) + "\n" for entry in entries])
    with open(path, "w") as csv:
        csv.writelines(lines)

if __name__ == "__main__":
    dataset_path = "../../regression-tree-benchmark/experiments/datasets/"
    datasets = []
    for (dirpath, _, filenames) in os.walk(dataset_path):
        for file in filenames:
            path = (dirpath, file)
            print(f"Found dataset: {path}")
            datasets.append(path)
    
    osrt_base = "./data/osrt"
    streed_base = "./data/streed"
    os.makedirs(osrt_base, exist_ok=True)
    os.makedirs(streed_base, exist_ok=True)

    for (dirpath, file) in datasets:
        dataset = parse_dataset(os.path.join(dirpath, file))
        write_osrt_dataset(dataset, os.path.join(osrt_base, file))
        write_streed_dataset(dataset, os.path.join(streed_base, file))

    with open("./data/datasets.txt", "w") as datasets_file:
        datasets_file.writelines([file + "\n" for (dirpath, file) in datasets])