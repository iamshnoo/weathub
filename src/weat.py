# inspired from https://github.com/McGill-NLP/bias-bench/tree/main implementation

import math
import random
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from encoding_utils import *
from tqdm import tqdm


class WEAT:
    def __init__(
        self,
        encode_function,
        target_set_1,
        target_set_2,
        attribute_set_1,
        attribute_set_2,
        num_partitions=100000,
        num_workers=None,
        normalize_test_statistic=False,
        encode_args=None,
        generate_bootstraps=True,
        bootstrap_size=5000,
        bootstrap_confidence=95,
    ):
        assert encode_function is not None, "encode_function cannot be None"
        assert encode_args is not None, "encode_args cannot be None"
        self.encode_function = encode_function
        self.encode_args = encode_args
        self.target_set_1 = target_set_1
        self.target_set_2 = target_set_2
        self.attribute_set_1 = attribute_set_1
        self.attribute_set_2 = attribute_set_2

        self.X = self.encode_function(self.target_set_1, self.encode_args)
        self.Y = self.encode_function(self.target_set_2, self.encode_args)
        self.A = self.encode_function(self.attribute_set_1, self.encode_args)
        self.B = self.encode_function(self.attribute_set_2, self.encode_args)

        self.all_cosine_similarities = self.precompute_cosine_similarities()
        self.s_word_AB_memoized = {
            index: self.s_word_AB(index) for index in range(len(self.X) + len(self.Y))
        }

        self.effect_size = self.compute_effect_size()
        self.do_bootstrap = generate_bootstraps
        if self.do_bootstrap:
            self.effect_size_ci, self.test_statistic_ci = self.bootstrap(
                n_samples=bootstrap_size,
                confidence=bootstrap_confidence,
            )
        (
            self.p_value,
            self.all_si,
            self.test_statistic,
        ) = self.p_value_permutation_test(
            num_partitions,
            num_workers,
            normalize_test_statistic,
        )

    def precompute_cosine_similarities(self, X=None, Y=None, A=None, B=None):
        if X is None or Y is None or A is None or B is None:
            XY = np.vstack((self.X, self.Y))
            AB = np.vstack((self.A, self.B))
            cosine_similarities = np.dot(XY, AB.T)
            # Add a small epsilon value to avoid division by zero
            epsilon = 1e-8
            cosine_similarities /= (
                np.outer(np.linalg.norm(XY, axis=1), np.linalg.norm(AB, axis=1))
                + epsilon
            )
            return cosine_similarities
        else:
            XY = np.vstack((X, Y))
            AB = np.vstack((A, B))
            cosine_similarities = np.dot(XY, AB.T)
            # Add a small epsilon value to avoid division by zero
            epsilon = 1e-8
            cosine_similarities /= (
                np.outer(np.linalg.norm(XY, axis=1), np.linalg.norm(AB, axis=1))
                + epsilon
            )
            return cosine_similarities

    def s_word_AB(self, index, A=None, cossims=None):
        if A is None or cossims is None:
            s_word_A = self.all_cosine_similarities[index, : len(self.A)].mean()
            s_word_B = self.all_cosine_similarities[index, len(self.A) :].mean()
            return s_word_A - s_word_B
        else:
            s_word_A = cossims[index, : len(A)].mean()
            s_word_B = cossims[index, len(A) :].mean()
            return s_word_A - s_word_B

    def s_set_AB(self, indices, s_word_AB_memoized=None, normalize=False):
        if s_word_AB_memoized is None:
            total_sum = sum(map(self.s_word_AB_memoized.__getitem__, indices))
        else:
            total_sum = sum(map(s_word_AB_memoized.__getitem__, indices))

        return total_sum / len(indices) if normalize and len(indices) > 0 else total_sum

    def s_sets_difference_AB(
        self, X_indices, Y_indices, s_word_AB_memoized=None, normalize=False
    ):
        return self.s_set_AB(X_indices, s_word_AB_memoized, normalize) - self.s_set_AB(
            Y_indices, s_word_AB_memoized, normalize
        )

    def compute_effect_size(self, X=None, Y=None, s_word_AB_memoized=None):
        if X is None:
            numerator = self.s_sets_difference_AB(
                range(len(self.X)),
                range(len(self.X), len(self.X) + len(self.Y)),
                s_word_AB_memoized=s_word_AB_memoized,
                normalize=True,
            )
            denominator = np.std(list(self.s_word_AB_memoized.values()), ddof=1)
        else:
            numerator = self.s_sets_difference_AB(
                range(len(X)),
                range(len(X), len(X) + len(Y)),
                s_word_AB_memoized=s_word_AB_memoized,
                normalize=True,
            )
            denominator = np.std(list(s_word_AB_memoized.values()), ddof=1)

        epsilon = 1e-8
        return numerator / (denominator + epsilon)

    def compute_bootstrap_effect_size_test_statistic(
        self, X_indices, Y_indices, A_indices, B_indices
    ):
        try:
            X = self.X[X_indices]
            Y = self.Y[Y_indices]
            A = self.A[A_indices]
            B = self.B[B_indices]
        except TypeError:
            X = np.array(self.X)[X_indices]
            Y = np.array(self.Y)[Y_indices]
            A = np.array(self.A)[A_indices]
            B = np.array(self.B)[B_indices]

        XY = np.vstack((X, Y))
        AB = np.vstack((A, B))
        cossims = np.dot(XY, AB.T)
        epsilon = 1e-8
        cossims /= (
            np.outer(np.linalg.norm(XY, axis=1), np.linalg.norm(AB, axis=1)) + epsilon
        )

        s_word_AB_memoized = {
            index: cossims[index, : len(A)].mean() - cossims[index, len(A) :].mean()
            for index in range(len(X) + len(Y))
        }

        term1 = sum(map(s_word_AB_memoized.__getitem__, range(len(X)))) / len(X)
        term2 = sum(
            map(s_word_AB_memoized.__getitem__, range(len(X), len(X) + len(Y)))
        ) / len(Y)
        numerator = term1 - term2
        denominator = np.std(list(s_word_AB_memoized.values()), ddof=1)
        epsilon = 1e-8
        effect_size = numerator / (denominator + epsilon)
        test_statistic = numerator
        return effect_size, test_statistic

    def bootstrap(self, n_samples=5000, confidence=95):
        effect_sizes = []
        test_statistics = []
        for _ in range(n_samples):
            indices_X1 = random.choices(range(len(self.X)), k=len(self.X))
            indices_Y1 = random.choices(range(len(self.Y)), k=len(self.Y))
            indices_A1 = random.choices(range(len(self.A)), k=len(self.A))
            indices_B1 = random.choices(range(len(self.B)), k=len(self.B))
            (
                effect_size,
                test_statistic,
            ) = self.compute_bootstrap_effect_size_test_statistic(
                indices_X1, indices_Y1, indices_A1, indices_B1
            )
            effect_sizes.append(effect_size)
            test_statistics.append(test_statistic)
        return np.percentile(
            effect_sizes, [(100 - confidence) / 2, (100 - (100 - confidence) / 2)]
        ), np.percentile(
            test_statistics, [(100 - confidence) / 2, (100 - (100 - confidence) / 2)]
        )

    def generate_partitions(self, num_partitions=10000):
        size = min(len(self.X), len(self.Y))
        combined_sets = np.vstack((self.X, self.Y))
        total_possible = math.comb(len(combined_sets), size)

        if num_partitions == "all" or num_partitions > total_possible:
            return self.generate_all_partitions(combined_sets, size)

        n = len(combined_sets)
        indices_set = set()

        while len(indices_set) < num_partitions:
            indices_set.update(
                tuple(sorted(np.random.choice(n, size, replace=False)))
                for _ in range(num_partitions - len(indices_set))
            )

        partitions = [
            (np.array(indices), np.setdiff1d(np.arange(len(combined_sets)), indices))
            if len(self.X) in [size, len(self.Y)]
            else (
                np.setdiff1d(np.arange(len(combined_sets)), indices),
                np.array(indices),
            )
            for indices in indices_set
        ]

        print(f"Generated {len(partitions)} partitions...")
        return partitions

    def generate_all_partitions(self, combined_sets, size):
        n = len(combined_sets)
        partitions = [
            (np.array(comb_indices), np.setdiff1d(np.arange(n), comb_indices))
            for comb_indices in combinations(range(n), size)
        ]
        print(f"Generated all possible, i.e {len(partitions)} partitions...")
        return partitions

    def process_partition(self, partition, test_statistic, normalize):
        Xi_indices, Yi_indices = partition
        # if normalize contains True, then we normalize the test statistic
        if normalize:
            si = self.s_sets_difference_AB(Xi_indices, Yi_indices, normalize=True)
        else:
            si = self.s_sets_difference_AB(Xi_indices, Yi_indices, normalize=False)
        return si, si > test_statistic

    def p_value_permutation_test(
        self,
        num_partitions=10000,
        num_workers=None,
        normalize=False,
    ):
        total_true = 0
        total = 0
        partitions = self.generate_partitions(num_partitions)
        if normalize:
            print("Normalizing test statistic...")
        test_statistic = self.s_sets_difference_AB(
            range(len(self.X)),
            range(len(self.X), len(self.X) + len(self.Y)),
            normalize=normalize,
        )

        all_si = []

        num_workers = num_workers or os.cpu_count()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                executor.map(
                    self.process_partition,
                    partitions,
                    [test_statistic] * len(partitions),
                    [normalize] * len(partitions),
                )
            )

            all_si = [si for si, _ in results]
            total_true = sum(is_true for _, is_true in results)
            total = len(partitions)

        return total_true / total, all_si, test_statistic

    def plot_test_statistic(self, all_si=None, test_statistic=None):
        all_si = all_si or self.all_si
        test_statistic = test_statistic or self.test_statistic

        # Plot the si values as blue points
        plt.scatter(range(len(all_si)), all_si, color="blue", label="s_i values")

        # Plot the test statistic as a horizontal red line
        plt.axhline(test_statistic, color="red", linestyle="--", label="Test Statistic")

        # Add labels and legend
        plt.xlabel("Partition Index")
        plt.ylabel("Test Statistic Value")
        plt.legend()

        plt.ylim(-2, 2)

        # Show the plot
        plt.show()


# DEMO usage
if __name__ == "__main__":
    np.random.seed(42)
    target_set_1 = ["physics", "chemistry"]
    target_set_2 = ["poetry", "drama", "dance"]
    attribute_set_1 = ["brother", "father"]
    attribute_set_2 = ["sister", "mother"]

    args = {
        "lang": "en",
        "embedding_type": "contextual",
        "encoding_method": "1",
        "subword_strategy": "first",
        "phrase_strategy": "average",
    }

    weat = WEAT(
        encode_function=encode_words,
        target_set_1=target_set_1,
        target_set_2=target_set_2,
        attribute_set_1=attribute_set_1,
        attribute_set_2=attribute_set_2,
        num_partitions=100000,  # "all",
        num_workers=None,
        normalize_test_statistic=True,
        encode_args=args,
        generate_bootstraps=True,
        bootstrap_size=2,
        bootstrap_confidence=95,
    )

    print("Effect size:", weat.effect_size)
    print("P value :", weat.p_value)
    print("Test statistic:", weat.test_statistic)
    print("-" * 50)
    weat.plot_test_statistic()
