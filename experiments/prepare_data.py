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
    datasets_large = []
    for (dirpath, _, filenames) in os.walk(dataset_path):
        for file in filenames:
            path = (dirpath, file)
            # Treat datasets >3MB different.
            if os.stat(os.path.join(dirpath, file)).st_size > 3 * 1024 * 1024:
                print(f"Found large dataset: {path}")
                datasets_large.append(path)
            else:
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

    osrt_base = "./data_large/osrt"
    streed_base = "./data_large/streed"
    os.makedirs(osrt_base, exist_ok=True)
    os.makedirs(streed_base, exist_ok=True)

    splits = []
    for (dirpath, file) in datasets_large:
        header, entries = parse_dataset(os.path.join(dirpath, file))
        i = 4
        while 10**i < len(entries):
            name = f"split_e{i}_{file}"
            splits.append(name)
            chunk = entries[0:10**i]
            write_osrt_dataset((header, chunk), os.path.join(osrt_base, name))
            write_streed_dataset((header, chunk), os.path.join(streed_base, name))
            i += 1
        name = "split_full_" + file
        splits.append(name)
        write_osrt_dataset((header, entries), os.path.join(osrt_base, name))
        write_streed_dataset((header, entries), os.path.join(streed_base, name))
    
    with open("./data/datasets_large.txt", "w") as datasets_file:
        datasets_file.writelines([file + "\n" for file in splits])