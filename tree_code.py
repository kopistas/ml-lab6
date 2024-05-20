import numpy as np
from collections import Counter

import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]

    thresholds = (sorted_features[:-1] + sorted_features[1:]) / 2
    unique_thresholds = np.unique(thresholds)

    def gini(targets):
        m = len(targets)
        if m == 0:
            return 0
        count = Counter(targets)
        p1 = count[1] / m
        p0 = count[0] / m
        return 1 - p1 ** 2 - p0 ** 2

    # Массивы для хранения значений Джини для каждого порога
    ginis = np.zeros(len(unique_thresholds))

    # Перебираем все пороги и вычисляем критерий Джини
    for i, threshold in enumerate(unique_thresholds):
        left_indices = sorted_features <= threshold
        right_indices = sorted_features > threshold

        left_targets = sorted_targets[left_indices]
        right_targets = sorted_targets[right_indices]

        n = len(target_vector)
        n_left = len(left_targets)
        n_right = len(right_targets)

        gini_left = gini(left_targets)
        gini_right = gini(right_targets)

        gini_split = (n_left / n) * gini_left + (n_right / n) * gini_right
        ginis[i] = gini_split

    gini_best = np.min(ginis)
    threshold_best = unique_thresholds[np.argmin(ginis)]

    return unique_thresholds, ginis, threshold_best, gini_best


import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if any(ft not in {"real", "categorical"} for ft in feature_types):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]) or (self._max_depth is not None and depth >= self._max_depth) or (
                self._min_samples_split is not None and len(sub_y) < self._min_samples_split):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, float('inf'), None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {
                    key: clicks.get(key, 0) / count for key, count in counts.items()
                }
                sorted_categories = sorted(ratio, key=ratio.get)
                categories_map = {
                    category: i for i, category in enumerate(sorted_categories)
                }
                feature_vector = np.vectorize(categories_map.get)(sub_X[:, feature])
            else:
                raise ValueError("Некорректный тип признака")

            if len(np.unique(feature_vector)) <= 1:
                continue

            thresholds, ginis, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini < gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = sorted_categories[:int(threshold + 1)]

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError("Некорректный тип признака")

        node["left_child"], node["right_child"] = {}, {}
        left_split = sub_X[:, feature_best] < threshold_best if self._feature_types[feature_best] == "real" else np.isin(sub_X[:, feature_best], threshold_best)
        right_split = ~left_split

        if np.sum(left_split) > 0:
            self._fit_node(sub_X[left_split], sub_y[left_split], node["left_child"], depth + 1)
        else:
            node["left_child"]["type"] = "terminal"
            node["left_child"]["class"] = Counter(sub_y).most_common(1)[0][0]

        if np.sum(right_split) > 0:
            self._fit_node(sub_X[right_split], sub_y[right_split], node["right_child"], depth + 1)
        else:
            node["right_child"]["type"] = "terminal"
            node["right_child"]["class"] = Counter(sub_y).most_common(1)[0][0]

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        if self._feature_types[node["feature_split"]] == "real":
            if x[node["feature_split"]] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif self._feature_types[node["feature_split"]] == "categorical":
            if x[node["feature_split"]] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError("Некорректный тип признака")

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_params(self, deep=True):
        return {"feature_types": self._feature_types, "max_depth": self._max_depth,
                "min_samples_split": self._min_samples_split, "min_samples_leaf": self._min_samples_leaf}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
