import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def percent_of_na(df):
    return (df.isna().sum()/df.shape[0]*100).sort_values(ascending=False)

def onehot(df, cols, type=int):
    result = df
    for col in cols:
        df_oh = pd.get_dummies(result[col], prefix=col, dtype=type)
        result = pd.concat([result, df_oh], axis=1)
        result.drop(columns=[col], inplace=True)
    return result

def heatmap(df):
    plt.figure(figsize=(20, 20))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Heatmap")
    plt.show()

def draw_scatter(df, threshold=0.6):
    correlation_matrix = df.corr()
    correlation_pairs = correlation_matrix.stack().reset_index()
    correlation_pairs.columns = ['Feature1', 'Feature2', 'Correlation']
    correlation_pairs = correlation_pairs[(correlation_pairs['Correlation'].abs() > threshold) & 
                                        (correlation_pairs['Feature1'] != correlation_pairs['Feature2'])]
    
    correlation_pairs['pair'] = correlation_pairs.apply(lambda row: tuple(sorted((row['Feature1'], row['Feature2']))), axis=1)
    correlation_pairs = correlation_pairs.drop_duplicates(subset=['pair'])
    correlation_pairs[['Feature1', 'Feature2']] = pd.DataFrame(correlation_pairs['pair'].tolist(), index=correlation_pairs.index)
    correlation_pairs = correlation_pairs.drop(columns=['pair'])

    for _, row in correlation_pairs.iterrows():
        feature1 = row['Feature1']
        feature2 = row['Feature2']
        plt.figure(figsize=(6, 4))
        sns.lmplot(x=feature1, y=feature2, data=df)
        plt.show()