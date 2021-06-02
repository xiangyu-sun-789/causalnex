import pandas as pd
from sklearn import preprocessing
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.structure import dynotears

# silence warnings
import warnings

warnings.filterwarnings("ignore")


def load_sports_data():
    data_file = "/Users/shawnxys/Development/Data/preprocessed_causal_sports_data_by_games/17071/features_shots_rewards.csv"

    features_shots_rewards_df = pd.read_csv(data_file)
    # rename column name
    features_shots_rewards_df = features_shots_rewards_df.rename(columns={'reward': 'goal'})

    X = features_shots_rewards_df.to_numpy()  # (number of time steps, number of variables)

    # data standardization
    scaler = preprocessing.StandardScaler().fit(X)
    normalized_X = scaler.transform(X)  # (number of time steps, number of variables)

    print('feature std after standardization: ', normalized_X.std(axis=0))
    assert (normalized_X.std(axis=0).round(
        decimals=3) == 1).all()  # make sure all the variances are (very close to) 1

    T, N = normalized_X.shape  # (number of time steps, number of variables)

    assert T == 4021 and N == 12

    # Initialize dataframe object, specify variable names
    variables_names = [s for s in features_shots_rewards_df.columns]

    return normalized_X, variables_names


if __name__ == "__main__":
    normalized_data_array, variables_names = load_sports_data()

    # convert numpy array to pandas dataframe
    normalized_data_df = pd.DataFrame(normalized_data_array, index=None, columns=variables_names)
    print(normalized_data_df.columns)

    # p: Number of past interactions we allow the model to create. The state of a variable at time `t` is
    # affected by past variables up to a `t-p`, as well as by other variables at `t`.
    estimated_DAG = dynotears.from_pandas_dynamic(normalized_data_df, p=1, h_tol=1e-60)

    # remove edges that are too weak
    estimated_DAG.remove_edges_below_threshold(0.01)

    # show edges with high weight coefficients thicker.
    edge_attributes = {
        (u, v): {
            "penwidth": w * 20 + 2,  # Setting edge thickness
            "arrowsize": 2 - 2.0 * w,  # Avoid too large arrows
        }
        for u, v, w in estimated_DAG.edges(data="weight")
    }

    viz = plot_structure(
        estimated_DAG,
        prog='dot',
        graph_attributes={"nodesep": 1.1},  # separation between nodes with same rank
        edge_attributes=edge_attributes,
        all_node_attributes=NODE_STYLE.WEAK,
        all_edge_attributes=EDGE_STYLE.WEAK)

    viz.draw('./estimated_DAG_sports.png')
