import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

df=pd.read_csv("F:\ML LAB EXP\Market_Basket_Optimisation.csv")
missing_values = df.isnull().sum()
print(missing_values)
import numpy as np
df.fillna(value=np.nan, inplace=True)
one_hot_encoded = pd.get_dummies(df.stack()).groupby(level=0).sum()
transactions_encoded = (one_hot_encoded > 0).astype(int)

import warnings
warnings.filterwarnings("ignore")
from mlxtend.frequent_patterns import apriori, association_rules
# Apriori algorithm
frequent_itemsets = apriori(transactions_encoded, min_support=0.02, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)


# Assume transactions_encoded is your DataFrame with the one-hot encoded data
# Run the Apriori algorithm to get frequent itemsets
frequent_itemsets = apriori(transactions_encoded, min_support=0.02, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)

# Filter for high confidence rules
high_conf_rules = rules[rules['confidence'] > 0.7]

# Extract the antecedents and consequents
high_conf_rules['antecedents'] = high_conf_rules['antecedents'].apply(lambda x: list(x)[0] if len(x) > 0 else x)
high_conf_rules['consequents'] = high_conf_rules['consequents'].apply(lambda x: list(x)[0] if len(x) > 0 else x)

# Create a directed graph from the rules
G = nx.from_pandas_edgelist(high_conf_rules, source='antecedents', target='consequents', edge_attr=True, create_using=nx.DiGraph())

# Plot the graph
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, seed=42)  # positions for all nodes
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', arrowsize=20)
edge_labels = nx.get_edge_attributes(G, 'confidence')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title('High Confidence Item Association Network')
plt.show()
