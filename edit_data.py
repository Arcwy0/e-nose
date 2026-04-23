import pandas

df = pandas.read_csv('database_robodog_new.csv')

for col in df.columns[:5]:
    print(col)
    df[col].values[:] = 0

df.to_csv('database_robodog_new_edit.csv')