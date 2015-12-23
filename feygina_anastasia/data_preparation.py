import pandas as pd


def read_csv(file_name):
    return pd.read_csv(file_name, sep=',')


def replace_nan_with_mean(dataframe):
    dataframe['Zytogen'].fillna(1, inplace=True)
    dataframe['Region'].fillna(77, inplace=True)
    return dataframe


def category_to_binary(df):
    df_new_type = df.drop('ID', axis=1)
    df_new_type['Age'] = pd.cut(df['Age'], 8, labels=[1, 2, 3, 4, 5, 6, 7, 8])
    df_new_type['Leuc'] = pd.cut(df['Leuc'], 2, labels=[1, 2])
    df_new_type['Leber'] = pd.cut(df['Leber'], 2, labels=[1, 2])
    df_new_type['weight'] = pd.cut(df['weight'], 3, labels=[1, 2, 3])
    df_new_type['height'] = pd.cut(df['height'], 3, labels=[1, 2, 3])
    target = df_new_type.ix[:, [-1]]
    df_new_type = df_new_type.drop('Better', axis=1)
    column_names = df_new_type.columns.values
    for col in column_names:
        df_new_type[col] = df_new_type[col].astype('category')
    df_binary = pd.get_dummies(df_new_type)
    df_binary['Better'] = pd.Series(target.values[:, 0], index=df_binary.index)
    df_binary.to_csv('fnkc_binary.csv', sep=',', mode='w')
    return df_binary


def divide_test_train(df_binary):
    test_df = df_binary.loc[df_binary['Better'] == 0]
    train_df = df_binary.loc[df_binary['Better'] != 0]
    return test_df, train_df


def split_the_data_into_classes(data):
    treat_300 = data.loc[data['Better'] == 300]
    treat_100 = data.loc[data['Better'] == 100]
    return treat_300, treat_100


def dataframe_to_string(dataframe):
    string = dataframe.values
    string = ['%.0f' % j for j in string]
    string = [''.join(string[0:-1]), string[-1]]
    # print(string)
    return string


def cross(line1, line2):
    k = len(line1)
    line1 = int(line1, 2)
    line2 = int(line2, 2)
    result = ("{0:0"+str(k)+"b}").format(line1 & line2)
    return result

