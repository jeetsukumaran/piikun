import numpy as np
from itertools import combinations

import numpy as np
from itertools import combinations

import numpy as np
from itertools import combinations

class AMOVA:
    def __init__(self, logger=None):
        self.logger = logger

    def log_info(self, message):
        if self.logger:
            self.logger.log_info(message)

    def log_warning(self, message):
        if self.logger:
            self.logger.log_warning(message)

    def log_error(self, message):
        if self.logger:
            self.logger.log_error(message)

    def _validate_groups(self, groups, n):
        indices = set(self._flatten(groups))
        if len(indices) != n or max(indices) >= n or min(indices) < 0:
            self.log_error("Group indices are invalid or do not match distance matrix size.")
            return False
        return True

    def _flatten(self, nested_list):
        for item in nested_list:
            if isinstance(item, list):
                yield from self._flatten(item)
            else:
                yield item

    def _pairwise_differences(self, distance_matrix, indices):
        if len(indices) > 1:
            return np.mean([distance_matrix[i, j] for i, j in combinations(indices, 2)])
        return 0

    def _ss_total(self, distance_matrix):
        mean_distance = np.mean(distance_matrix)
        return np.sum((distance_matrix - mean_distance) ** 2)

    def _ss_among_groups(self, distance_matrix, groups):
        overall_mean = np.mean(distance_matrix)
        ss_among = sum(len(list(self._flatten(group))) * (self._pairwise_differences(distance_matrix, list(self._flatten(group))) - overall_mean) ** 2 for group in groups)
        return ss_among

    def _ss_within_groups(self, distance_matrix, groups):
        ss_within = sum(self._ss_total(distance_matrix[list(self._flatten(group)), :][:, list(self._flatten(group))]) for group in groups)
        return ss_within

    def _phi(self, ss_among, ss_within, ss_total):
        if ss_total == 0:
            return 0
        return ss_among / (ss_total - ss_within)

    def amova(self, distance_matrix, groups, validate=True):
        if validate and not self._validate_groups(groups, distance_matrix.shape[0]):
            return

        ss_total = self._ss_total(distance_matrix)
        ss_among_groups = self._ss_among_groups(distance_matrix, groups)
        ss_within_groups = self._ss_within_groups(distance_matrix, groups)
        phi_st = self._phi(ss_among_groups, ss_within_groups, ss_total)

        results = {
            'SSD_Total': ss_total,
            'SSD_Among_Groups': ss_among_groups,
            'SSD_Within_Groups': ss_within_groups,
            'PhiST': phi_st,
            'Detailed_Phi': {}
        }

        detailed_phi = {}
        for i, group in enumerate(groups):
            if any(isinstance(subgroup, list) for subgroup in group):  # Modified check for nested groups
                subgroup_results = self.amova(distance_matrix, group, validate=False)
                sg_results = subgroup_results.get('PhiST', 0)
                detailed_phi[f'Group_{i}'] = sg_results

        if detailed_phi:  # Only add to results if detailed_phi is not empty
            results['Detailed_Phi'] = detailed_phi

        self.log_info("AMOVA analysis completed successfully.")
        return results

# Example usage of the updated class here



# # Define a simple logger for demonstration purposes
# class SimpleLogger:
#     def log_info(self, message):
#         print(f"INFO: {message}")

#     def log_warning(self, message):
#         print(f"WARNING: {message}")

#     def log_error(self, message):
#         print(f"ERROR: {message}")

# # Instantiate the AMOVA class with a simple logger
# amova_analysis = AMOVA(logger=SimpleLogger())

# # Define a 9x9 distance matrix for demonstration
# distance_matrix = np.array([
#     [0, 1, 1, 2, 2, 2, 3, 3, 3],
#     [1, 0, 1, 2, 2, 2, 3, 3, 3],
#     [1, 1, 0, 2, 2, 2, 3, 3, 3],
#     [2, 2, 2, 0, 1, 1, 3, 3, 3],
#     [2, 2, 2, 1, 0, 1, 3, 3, 3],
#     [2, 2, 2, 1, 1, 0, 3, 3, 3],
#     [3, 3, 3, 3, 3, 3, 0, 1, 1],
#     [3, 3, 3, 3, 3, 3, 1, 0, 1],
#     [3, 3, 3, 3, 3, 3, 1, 1, 0],
# ])

# # Group specification with 3 nested levels
# # Level 1: Two main groups
# # Level 2: Each main group has 2 subgroups
# # Level 3: Each subgroup has individual elements
# groups = [
#     [   # First main group
#         [[0], [1]],  # Subgroups
#         [[2], [3]]   # Subgroups
#     ],
#     [   # Second main group
#         [[4], [5]],  # Subgroups
#         [[6], [7], [8]]  # Subgroups with 3 elements
#     ]
# ]

# # Run the AMOVA analysis
# results = amova_analysis.amova(distance_matrix, groups, validate=True)

# # Display the results
# print(results)

class SampleStratifiedVariation:
    def __init__(self, rng=None, seed=None):
        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng(seed)

    def generate_distance_matrix(self, grouping, within_variance=1.0, between_variance=2.0):
        """
        Generates a distance matrix based on the specified grouping, within-group variance,
        and between-group variance.

        Parameters:
        - grouping: A list of lists indicating group memberships. E.g., [[0, 1], [2, 3, 4]]
        - within_variance: Variance of distances within each group.
        - between_variance: Variance of distances between groups.

        Returns:
        A numpy array representing the distance matrix.
        """
        total_elements = sum(len(subgroup) for subgroup in grouping)
        distance_matrix = np.zeros((total_elements, total_elements))

        # Fill within-group distances
        for group in grouping:
            for i in group:
                for j in group:
                    if i == j:
                        distance_matrix[i, j] = 0
                    else:
                        distance_matrix[i, j] = self.rng.normal(0, within_variance)

        # Calculate mean distances for between-group variance
        mean_distance = np.mean([distance_matrix[i, j] for group in grouping for i in group for j in group if i != j])

        # Adjust distances to incorporate between-group variance
        for i in range(total_elements):
            for j in range(total_elements):
                if not any(i in group and j in group for group in grouping):
                    # Increase the distance to simulate between-group variance
                    adjustment = self.rng.normal(mean_distance, between_variance)
                    distance_matrix[i, j] += adjustment
                    distance_matrix[j, i] += adjustment

        return distance_matrix

rng = np.random.default_rng(seed=42)  # For reproducibility
simulator = SampleStratifiedVariation(rng=rng)
grouping = [[0, 1], [2, 3], [4, 5, 6]]
distance_matrix = simulator.generate_distance_matrix(grouping)
print(distance_matrix)
amova_analysis = AMOVA()
results = amova_analysis.amova(distance_matrix, grouping, validate=True)
print(results)
