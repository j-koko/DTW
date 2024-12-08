import numpy as np
import csv
from s5834643 import dtw_match, dtw

def compare_results(dtw_results):
    """Determining dtw_match accuracy based on provided metadata"""
    with open("results.csv", mode="r") as file:
        metadata = csv.reader(file, delimiter='\t')
        metadata_list = [row for row in metadata]
        metadata_dict = {}
        for item in metadata_list:
            file_stripped = item[0].replace("'", "") # remove redundant quotes from metadata
            wake_stripped = item[1].replace("'", "") # remove redundant quotes from metadata
            metadata_dict[file_stripped] = wake_stripped

        dtw_results = dict(sorted(dtw_results.items())) # sort dicts to compare them against each other
        metadata_dict = dict(sorted(metadata_dict.items()))

        meta_data_matches = []
        for file, dtw_call in dtw_results.items():
            meta_call = metadata_dict.get(file)  # Get corresponding metadata value
            if meta_call is not None:
                match = (dtw_call == meta_call)
                meta_data_matches.append(match)
            else:
                # If the file is missing in metadata, count as mismatch
                meta_data_matches.append(False)

        # calculating accuracy rate (final rate: 97.5%)
        correct_matches = [match for match in meta_data_matches if match is True]
        accuracy = (len(correct_matches) / len(meta_data_matches)) * 100
        print(accuracy)

# print(dtw_match())
# compare_results(dtw_match())

template_1 = [1, 2, 4, 3, 2, 3]
test_1 = [2, 3, 1, 2]

template = np.array([[1, 2], [4, 3], [2, 3]])
test = np.array([[2, 3], [1, 3], [3, 1], [4, 2], [5, 2]])
print(dtw(template, test))
print(dtw(template_1, test_1))

print(dtw_match())
