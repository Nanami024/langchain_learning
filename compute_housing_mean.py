import pandas as pd
print(pd.read_csv("sop_knowledge/housing.csv")["MedHouseVal"].mean())
