from causalml.dataset import make_uplift_classification
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from uplift.meta_learner import NeVOX


def main():
    df, X_names = make_uplift_classification(
        n_samples=1000, treatment_name=["control", "treatment1", "treatment2"]
    )

    y = "conversion"
    T = "treatment_group_key"
    control = 0
    X = [
        col
        for col in df.columns
        if col not in [y, "conversion"] and "informative" in col
    ]

    T_encoder = OrdinalEncoder()
    df[T] = T_encoder.fit_transform(df[T].to_numpy().reshape(-1, 1)).astype(int)

    X = df[X].to_numpy()
    T = df[T].to_numpy()
    y = df[y].to_numpy()

    ic_lookup = {0: 0, 1: 0.1, 2: 0.1}
    cc_lookup = {0: 0, 1: 5, 2: 10}
    value = np.ones(df.shape[0]) * 20

    clf = NeVOX(ic_lookup=ic_lookup, cc_lookup=cc_lookup)
    clf.fit(X, y, T, value)



if __name__ == "__main__":
    main()
