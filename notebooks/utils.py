import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy.spatial import distance_matrix

from nteprsm.constants import NTEP_COLOR_SCALE


def set_custom_template():
    """
    Initializes and configures a custom Plotly template based on the "plotly_white" theme.
    The customizations include setting specific dimensions, grid colors, line visibility,
    tick positioning, and mirroring settings for the axes.

    This function modifies the global pio.templates object by adding a new template named 'custom'
    and sets it as the default template for all future plots.
    """
    # Clone an existing template
    custom_template = pio.templates["plotly_white"].to_plotly_json()

    # Modify the template
    custom_template["layout"]["width"] = 800
    custom_template["layout"]["height"] = 600
    custom_template["layout"]["yaxis"]["gridcolor"] = "white"
    custom_template["layout"]["yaxis"]["showline"] = True
    custom_template["layout"]["yaxis"]["linecolor"] = "black"
    custom_template["layout"]["yaxis"]["ticks"] = "outside"
    custom_template["layout"]["yaxis"]["mirror"] = True
    custom_template["layout"]["xaxis"]["gridcolor"] = "white"
    custom_template["layout"]["xaxis"]["showline"] = True
    custom_template["layout"]["xaxis"]["linecolor"] = "black"
    custom_template["layout"]["xaxis"]["mirror"] = True
    custom_template["layout"]["xaxis"]["ticks"] = "outside"

    # Save the custom template
    pio.templates["custom"] = custom_template
    pio.templates.default = "custom"


def softmax(z):
    """
    The softmax function takes as input a vector z of K real numbers,
    and normalizes it into a probability distribution consisting of K
    probabilities proportional to the exponentials of the input numbers.

    Args:
        vector (np.array or list): a vector of K real numbers

    Returns:
        np.array or list: a vector of K probabilities
    """
    e = np.exp(z)
    return e / e.sum()


def distmatrix_to_expquadkernel(dist_matrix, alpha, inv_rho, sigma_e):
    """"
    transform a distance matrix to exponential quadratic kernel for gaussian
    process
    k(x_i, x_j) = alpha^2 * exp(-0.5* (Dist_ij * inv_rho)^2) + \
                  delta_{i,j}*sigma_e^2

    Args:
        dist_matrix:     Eucleadian distance matrix calculated from field layout
        alpha:           marginal standard deviation, determines the average 
                         distance from the mean
        inv_rho:         inverse of the length parameter, determines the length 
                         of the 'wiggles' of the function
        sigma_e:         the addition of σ2 on the diagonal is important to 
                         ensure the positive definiteness of the resulting 
                         matrix in the case of two identical inputs. 
                         In statistical terms,  σ is the scale of the noise term 
                         in the regression. 

    Returns:
        kernel:          exponential quadratic kernel from distance matrix
    """
    kernel = alpha**2 * np.exp(-0.5 * np.square(dist_matrix * inv_rho))
    np.fill_diagonal(kernel, kernel.diagonal() + sigma_e**2)
    return kernel


def transform_layout_to_long_format(
    layout: pd.DataFrame, value_name: str
) -> pd.DataFrame:
    """
    transform layout that is orgranized by row and column to long data format

    Args:
        layout (pd.DataFrame): field layout indexed by rows # and cols by col #
        value_name(str)      : value name in the long data

    Returns:
        pd.DataFrame:          data frame in long format with ROW and COl
    """
    layout.index.name = "ROW"
    layout = layout.reset_index().melt(
        id_vars="ROW", var_name="COL", value_name=value_name
    )
    return layout


def simulate_rating_scores(theta, beta, taus):
    """
    Simulate rating scores for a give theta value, rating severity (beta) and
    rating thresholds (tau)

    Args:
        theta (float):             the perceived turf quality
        beta (float):              rating severity
        taus (np.array or list ):  category thresholds

    Returns:
        rating (int):              simulated rating score
    """
    unsumed = theta - beta - taus
    unsumed = np.insert(unsumed, 0, 0)
    numerators = np.exp(np.cumsum(unsumed))
    denominators = sum(numerators)
    probs = numerators / denominators
    rating = np.random.choice(range(1, len(taus) + 2), p=probs)
    return rating


def get_model_data(data, **kwargs):
    """
    get model_data for stan from ntep data.

    Args:
       data(pd.DataFrame):        raw NTEP rating data

    Returns:
       model_data(dict):          model_data in format of a dictionary
    """

    y = np.asarray(data.QUALITY - data.QUALITY.min())  # y start from one
    ii = np.asarray(data.RATING_EVENT_CODE)
    jj = np.asarray(data.ENTRY_CODE)
    pp = np.asarray(data.PLOC_CODE)
    # tt = np.asarray(data.YEAR_CODE)
    # kk = np.asarray(data.MONTH_CODE)
    N = len(y)
    M = len(np.unique(y)) - 1
    I = max(ii)
    J = max(jj)
    P = max(pp)
    # T = max(tt)
    # K = max(kk)
    model_data = {
        "y": y,
        "ii": ii,
        "jj": jj,
        "pp": pp,
        "N": N,
        "M": M,
        "I": I,
        "J": J,
        "P": P,
    }
    for key, value in kwargs.items():
        model_data[key] = value
    return model_data


def plot_ratings_over_time(data, plot_dims=[5.5, 5.5]):
    """
    plot rating data over time
    Args:
        data (dataframe): data (columns: row, col, date and rating)
        plot_dims (list, optional): _description_. Defaults to [5.5, 5.5].

    Returns:
        plotly figure object
    """
    dfs = [
        data[data.DATE == date].reset_index(drop=True) for date in data.DATE.unique()
    ]
    # generate the frames. NB name
    frames = [
        go.Frame(
            data=go.Heatmap(
                z=df.QUALITY,
                y=df.ROW,
                x=df.COL,
                hoverinfo="text",
                text=[
                    "Entry:" + name + "<br />Quality:" + str(quality)
                    for name, quality in zip(df.ENTRY_NAME, df.QUALITY)
                ],
                colorscale=[NTEP_COLOR_SCALE[i] for i in np.sort(df.QUALITY.unique())],
            ),
            name=df.DATE[0].strftime("%b, %Y"),
        )
        for df in dfs
    ]

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True}}],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "type": "buttons",
            }
        ],
        # iterate over frames to generate steps... NB frame name...
        sliders=[
            {
                "steps": [
                    {
                        "args": [
                            [f.name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": f.name,
                        "method": "animate",
                    }
                    for f in frames
                ],
            }
        ],
        width=10 * plot_dims[0] * data.COL.max(),
        height=10 * plot_dims[1] * data.ROW.max(),
        yaxis={"title": "Row #", "tick0": 1, "dtick": 1},
        xaxis={"title": "Col #", "tickangle": 0, "side": "top", "tick0": 1, "dtick": 1},
        title_x=0.5,
    )
    return fig


def plot_field_heat_map(
    plt_effect, row_id, col_id, entry_name, fig_width=400, fig_height=600
):
    fig6 = go.Figure()
    fig6.add_traces(
        go.Heatmap(
            z=plt_effect,
            y=row_id,
            x=col_id,
            hoverinfo="text",
            text=["Entry:" + name for name in entry_name],
        )
    )
    fig6.update_layout(
        width=fig_width,
        height=fig_height,
        title="Plot Location Effect",
        yaxis={"title": "Row #", "tick0": 1, "dtick": 1},
        xaxis={
            "title": "Col #",
            "tickangle": 0,
            "side": "bottom",
            "tick0": 1,
            "dtick": 1,
        },
    )
    fig6.show()


def calc_dist_matrix(
    layout_data, index_col="PLOC_CODE", row_len=5.5, col_len=5.5, row_gap=0, col_gap=0
):
    """
    layout data with three columns:
    SERP_ID: serialized plot id
    ROW: row id
    COL: col id
    index_col: index column
    row_len: row length
    col_len: column length
    row_gap: row gap length
    col_gap: column gap length
    given plot layout_data calculate distance matrix
    """
    layout_data = layout_data.set_index(index_col)
    layout_data.sort_index(inplace=True)
    layout_data = layout_data[["ROW", "COL"]]
    # rescale row and col coords based on plot dimension
    layout_data = layout_data.assign(
        ROW=layout_data.ROW * (col_len + col_gap) / (row_len + row_gap)
    )
    coords = layout_data.values
    dist_matrix = distance_matrix(coords, coords)
    dist_matrix = dist_matrix.astype(float)
    return dist_matrix


def subset_target_data(target, ntep_data, col_name="QUALITY"):
    """subset target data from all ratings"""
    ntep_qual = ntep_data[ntep_data.TRAIT.str.startswith(target)]
    ntep_qual = ntep_qual.astype({"VALUE": int})
    ntep_qual.rename(columns={"VALUE": col_name}, inplace=True)
    return ntep_qual


def get_entry_id2name_map(ntep_data):
    id_to_name = ntep_data.groupby("ENTRY_ID").apply(
        lambda x: x["ENTRY_NAME"].unique().tolist()
    )
    ids_to_save = id_to_name[id_to_name.apply(lambda x: len(x) == 1)]
    ids_1 = ids_to_save.apply(lambda x: x[0].strip())
    ids_to_check = id_to_name[id_to_name.apply(lambda x: len(x) > 1)]
    ids_2 = ids_to_check.apply(select_cultivar_name)
    id_to_name = pd.concat([ids_1, ids_2]).sort_index().to_dict()
    return id_to_name


def select_cultivar_name(name_list):
    name_list = [x.strip() for x in name_list]
    string_lens = [len(x) for x in name_list]

    shortest_name = name_list[string_lens.index(min(string_lens))]
    name_list.remove(shortest_name)

    name_list_new = [x for x in name_list if shortest_name.lower() in x.lower()]
    if len(name_list_new) >= 1:
        final_name = name_list_new[0]
    else:
        final_name = shortest_name
    return final_name


def extract(variable, results, n_burnin=400):
    col_name = variable.upper() + "_EFF"
    var_name = variable.upper() + "_CODE"
    str_to_drop = variable + "."
    results_data = results[results.index > n_burnin]
    variable_data = pd.DataFrame(
        results_data.loc[
            :,
            (results_data.columns.str.startswith(variable))
            & (~results_data.columns.str.contains("free")),
        ].mean(),
        columns=[col_name],
    )
    try:
        variable_data[var_name] = variable_data.index.str.replace(
            str_to_drop, ""
        ).astype(int)
    except TypeError:
        print("Warning: Typing Failed...")
    variable_data.reset_index(drop=True, inplace=True)
    return variable_data


def plot_thresholds(kappa):
    """
    plot estimated thresholds

    Args:
        kappa (pd.Series): thresholds
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=kappa, y=[0] * 8, mode="markers", marker_size=15, hoverinfo="x")
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(
        showgrid=False,
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=3,
        showticklabels=False,
    )
    fig.update_layout(height=200, plot_bgcolor="white")
    fig.show()


def predict_rating_histogram(theta, kappa):
    unsummed = np.array([theta] * 9) - np.array([0] + list(kappa))
    summed = np.cumsum(unsummed)
    px.bar(softmax(summed))


# load data from individual chain
def load_data_from_chain_csvs(path_to_file="", n_chain=4):
    results = []
    for i in range(n_chain):
        chain = pd.read_csv(path_to_file[:-5] + f"{i+1}.csv", comment="#")
        chain = chain.assign(chain=f"{i+1}")
        results.append(chain)
    results = pd.concat(results, ignore_index=False)
    return results


def plot_effect_size(results_data):
    var_data = (
        results_data.loc[:, (~results_data.columns.str.contains("free"))]
        .filter(regex="year|month|entry|beta|plot")
        .mean()
        .reset_index()
    )
    var_data[["VAR", "CODE"]] = var_data["index"].str.split(".", expand=True)
    var_data.rename(columns={0: "VALUE"}, inplace=True)
    fig = px.box(
        var_data,
        y="VAR",
        x="VALUE",
        category_orders={"VAR": ["month", "year", "plot", "entry", "beta"]},
        labels={"VALUE": "Effect Size on Logit Scale", "VAR": ""},
    )
    fig.update_layout(
        yaxis=dict(
            tickmode="array",
            tickvals=["plot", "beta", "month", "year", "entry"],
            ticktext=[
                "Plot Location Effect",
                "Rating Severity",
                "Month Effect",
                "Year Effect",
                "Entry Quality",
            ],
        )
    )
    fig.show()
