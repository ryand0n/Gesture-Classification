import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import linregress
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def convert(x):
    listed = x[1:-1].split(', ')
    return [eval(i) for i in listed]

def pca(x):
    X = []
    for index, val in enumerate(x):
        X.append([index, val])
    X = np.array(X)

    pca = PCA(n_components=2)
    pca.fit_transform(X)
    z = pca.singular_values_
    return z[1]

def calc_mean(df, col):
    return df[col].apply(lambda x: np.mean(x))

def calc_max(df, col):
    return df[col].apply(lambda x: np.max(x))

def calc_min(df, col):
    return df[col].apply(lambda x: np.min(x))

def get_slope(x):
    a = np.arange(len(x))
    b = x
    slope, intercept, r, p, se = linregress(a, b)
    return slope

def transform(df):
    df = df.drop(index=df.index[0]).reset_index().drop(columns=['index', 'level_0'])

    for col in df.columns:
        if col != 'target':
         df[col] = df[col].apply(lambda x: convert(x))

    og_cols = ['x', 'y', 'z']
    for col in df.columns:
        if col in og_cols:
            new_col = "{}_mean".format(col)
            df[new_col] = calc_mean(df, col)

    for col in df.columns:
        if col in og_cols:
            new_col = "{}_max".format(col)
            df[new_col] = calc_max(df, col)

    for col in df.columns:
        if col in og_cols:
            new_col = "{}_min".format(col)
            df[new_col] = calc_min(df, col)

    for col in df.columns:
        if col in og_cols:
            new_col = "{}_pca".format(col)
            df[new_col] = df[col].apply(lambda x: pca(x))

    for col in df.columns:
        if col in og_cols:
            new_col = "{}_slope".format(col)
            df[new_col] = df[col].apply(lambda x: get_slope(x))

    df = df.drop(columns=['x', 'y', 'z'])
    #df['target'] = pd.Series([1] * 49)
    return df

def transform_raw(dic):
    """
    dic: Dictionary with x, y, z as keys and lists of floats as values
    """
    data = {}
    index = ['x_mean', 'y_mean', 'z_mean', 'x_max', 'y_max', 'z_max', 'x_min',
       'y_min', 'z_min', 'x_pca', 'y_pca', 'z_pca', 'x_slope', 'y_slope',
       'z_slope']

    # creating features
    for key, value in dic.items():
        if key != 'index':
            mean = "{}_mean".format(key)
            data[mean] = np.mean(value)

            max = "{}_max".format(key)
            data[max] = np.max(value)

            min = "{}_min".format(key)
            data[min] = np.mean(value)

            pca_ = "{}_pca".format(key)
            data[pca_] = pca(value)

            slope = "{}_slope".format(key)
            data[slope] = get_slope(value)

    return pd.Series(data).reindex(index=index).to_numpy().reshape(1, -1)

def generate_random_data(df):
    random_data = np.random.rand(df.shape[0], df.shape[1])
    rand_df = pd.DataFrame(random_data, columns=df.columns)
    final = pd.concat([df, rand_df], axis=0)
    final.reset_index(inplace=True, drop=True)
    final['target'] = pd.Series([1] * df.shape[0] + [0] * df.shape[0])
    return final


def fit(df):
    X = df.loc[:, df.columns != 'target']
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"Test accuracy: {acc}")

    return clf

def predict(model, data):
    """
    data: Dictionary with x, y, z as keys and lists of floats as values
    model: LR Model that has already been trained
    """
    data = transform_raw(data)
    pred = model.predict(data)

    return pred

if __name__ == '__main__':
    raising = pd.read_csv("../data/raised_hand_data.csv")
    X = transform(raising)
    X = generate_random_data(X)
    print(X)
    model = fit(X)

    raise_data = [{'y': [0.0859375, 0.267334, 0.1723633, 0.5603027, 0.7158203, 0.7834473, 0.8330078, 0.8979492, 0.8972168, 0.9128418, 0.9020996, 0.8833008, 0.8161621, 0.6821289, 0.4240723, 0.104248, -0.1328125, -0.2404785, -0.002197266], 'x': [1.185059, 1.049805, 1.119873, 0.7885742, 0.6872559, 0.4438477, 0.1669922, 0.06005859, -0.02416992, -0.04614258, -0.05810547, -0.06640625, 0.1740723, 0.5432129, 0.9885254, 1.175537, 1.22168, 1.088623, 1.131836], 'index': 0, 'z': [-0.1933594, -0.1555176, -0.2121582, -0.3725586, -0.4265137, -0.4372559, -0.328125, -0.3276367, -0.411377, -0.4086914, -0.388916, -0.4255371, -0.2910156, -0.2724609, -0.2731934, -0.2331543, -0.2680664, -0.09204102, -0.04516602]}, {'y': [0.05151367, 0.5732422, 0.7312012, 0.8215332, 0.8618164, 0.9438477, 0.9216309, 0.9633789, 0.9335938, 0.9033203, 0.9160156, 0.918457, 0.9367676, 0.8422852, 0.7414551, 0.4880371, 0.2797852], 'x': [1.121094, 0.8188477, 0.5981445, 0.4929199, 0.3688965, 0.03271484, 0.1154785, 0.1447754, 0.02783203, 0.03198242, 0.005615234, -0.008056641, 0.0390625, 0.229248, 0.5639648, 0.9052734, 1.107666], 'index': 1, 'z': [-0.2441406, -0.2468262, -0.2841797, -0.2434082, -0.2145996, -0.3010254, -0.3596191, -0.3425293, -0.3271484, -0.3225098, -0.3422852, -0.2937012, -0.3383789, -0.3200684, -0.2182617, -0.1845703, -0.2919922]}, {'y': [0.5100098, -0.01171875, 0.1645508, 0.4001465, 0.5651855, 0.6794434, 0.7990723, 0.8803711, 0.8574219, 0.861084, 0.8442383, 0.8754883, 0.8847656, 0.8701172, 0.8078613, 0.6074219, 0.4460449], 'x': [1.883301, 1.112549, 0.9592285, 0.8913574, 0.7980957, 0.7299805, 0.5686035, 0.3310547, 0.1320801, -0.09521484, -0.1481934, -0.246582, -0.2792969, 0.09301758, 0.4404297, 0.7539063, 1.12793], 'index': 2, 'z': [-0.534668, -0.06933594, -0.4445801, -0.4672852, -0.4970703, -0.3820801, -0.3439941, -0.3583984, -0.3884277, -0.3098145, -0.3518066, -0.359375, -0.402832, -0.2487793, -0.2866211, -0.2836914, -0.1518555]}, {'y': [0.03662109, -0.2299805, -0.01708984, -0.003417969, 0.5839844, 0.7565918, 0.8417969, 0.9128418, 0.9453125, 0.9223633, 0.8972168, 0.8930664, 0.9077148, 0.8747559, 0.7573242, 0.5581055, 0.2915039, 0.06982422, -0.128418, 0.01171875, -0.006591797], 'x': [1.401611, 1.158936, 1.112549, 1.19751, 0.8225098, 0.6169434, 0.3103027, 0.1809082, 0.08520508, -0.06225586, -0.1147461, -0.1748047, -0.1474609, 0.1337891, 0.4916992, 0.800293, 1.033691, 1.271973, 1.188721, 1.196777, 1.112061], 'index': 3, 'z': [-0.2807617, -0.1882324, -0.02319336, -0.2336426, -0.2319336, -0.2128906, -0.2185059, -0.251709, -0.2453613, -0.3476563, -0.3276367, -0.3132324, -0.3271484, -0.2609863, -0.2199707, -0.1921387, -0.09204102, -0.1777344, -0.2097168, -0.1137695, -0.0534668]}, {'y': [0.03417969, 0.4248047, 0.6313477, 0.7380371, 0.8364258, 0.909668, 0.8793945, 0.8991699, 0.8955078, 0.8769531, 0.895752, 0.7521973, 0.5852051, 0.395752], 'x': [1.162598, 1.015381, 0.7512207, 0.6242676, 0.3950195, 0.2023926, 0.1105957, -0.1826172, -0.2341309, -0.248291, -0.1660156, 0.1787109, 0.7722168, 1.106934], 'index': 4, 'z': [-0.208252, -0.2570801, -0.2495117, -0.2675781, -0.222168, -0.2373047, -0.376709, -0.3027344, -0.3308105, -0.3325195, -0.3232422, -0.2851563, -0.175293, -0.2036133]}]
    for data in raise_data:
        print(predict(model, data))



