from sklearn.base import BaseEstimator, TransformerMixin

sensorColumns = ["dpFilter", "rms_vibration"]
intColumns = ["firmware_version", "dpFilter_present", "rms_vibration_present", "duty_class"]


def add_present_indicators(df):
    df = df.copy()
    for column in sensorColumns:
        df[column + "_present"] = df[column].notna()
    return df


def to_model_format(df):
    df = df.copy()
    version = df["firmware_version"].str.lstrip("v").str.split(".", expand=True).astype(int)
    df["firmware_version"] = version[0] * 100 + version[1] * 10 + version[2]
    df[intColumns] = df[intColumns].astype(int)
    return df


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, target_col):
        self.group_col = group_col
        self.target_col = target_col

    def fit(self, X, y=None):
        self.medians_ = X.groupby(self.group_col)[self.target_col].median()
        self.overall_ = X[self.target_col].median()
        return self

    def transform(self, X):
        X = X.copy()
        missing_mask = X[self.target_col].isna()
        if missing_mask.any():
            X.loc[missing_mask, self.target_col] = (
                X.loc[missing_mask, self.group_col].map(self.medians_).fillna(self.overall_)
            )
        return X
