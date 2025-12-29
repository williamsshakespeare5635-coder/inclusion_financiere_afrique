import pandas as pd

# Charger les données
df = pd.read_csv("Financial_inclusion_dataset.csv")

# Afficher les premières lignes
print(df.head())

# Informations générales
print(df.info())

# Statistiques descriptives
print(df.describe(include="all"))


from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="Financial Inclusion Report", explorative=True)
profile.to_file("profiling_report.html")
