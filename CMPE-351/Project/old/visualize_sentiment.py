import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = {
    'Review': [
        'Terrible service, but the food was good enough, not the best place for a quick meal though',
        'It was a nice place, but the food was only okay',
        'It was a nice place, but the food was okay',
        'It was a nice place, but the food was horrible'
    ],
    'Price': [32.45, 52.64, 75.04, 34.40],
    'Service': [-98.38, 61.15, 66.49, 29.64],
    'Ambiance': [49.74, 50.84, 56.64, 29.90],
    'Food': [99.09, -77.17, -77.92, -98.71]
}

# Convert data into a DataFrame
df = pd.DataFrame(data)

# Melting the DataFrame to make it suitable for seaborn's function
df_melted = df.melt(id_vars=["Review"], value_vars=["Price", "Service", "Ambiance", "Food"],
                    var_name="Aspect", value_name="Sentiment Score")

# Creating violin plots
plt.figure(figsize=(10, 6))
sns.violinplot(x="Aspect", y="Sentiment Score", data=df_melted, cut=0)
plt.title('Distribution of Sentiment Scores by Aspect')
plt.xlabel('Aspect')
plt.ylabel('Sentiment Score')
plt.grid(True)
plt.ylim(-100, 100) 
plt.show()

# Creating box plots
plt.figure(figsize=(10, 6))
sns.boxplot(x="Aspect", y="Sentiment Score", data=df_melted)
plt.title('Box Plot of Sentiment Scores by Aspect')
plt.xlabel('Aspect')
plt.ylabel('Sentiment Score')
plt.grid(True)
plt.show()
