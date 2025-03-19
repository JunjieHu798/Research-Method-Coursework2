import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# Load dataset
da = pd.read_csv('diet_data.csv', encoding='utf-8')
print("Raw data:")
print(da.head())

# Keep only variables related to environmental impact
elements = ["diet_group", "sex", "age_group",
            "mean_ghgs", "mean_land", "mean_watscar", "mean_bio", "mean_watuse"]
da = da[elements]
print("\nData after retaining relevant features:")
print(da.head())

# Normalize environmental indicators (to unify feature scales for better visualization)
Scaler = MinMaxScaler()
envir_vars = [col for col in da.columns if col.startswith("mean_")]
da[envir_vars] = Scaler.fit_transform(da[envir_vars])
print("\nNormalized environmental indicators:")
print(da[envir_vars].head())

# Encode diet_group (for color mapping in plotly)
diet_tags = da['diet_group'].unique()
print("\ndiet_group category:", diet_tags)

diet_code = {label: idx for idx, label in enumerate(diet_tags)}
print("diet_group encoding mapping:", diet_code)

da['diet_code'] = da['diet_group'].map(diet_code)
print("\nData after adding diet_code:")
print(da[['diet_group', 'diet_code']].head())

# Improve axis labels with abbreviations (for cleaner plots)
labels_cha = {
    "mean_ghgs": "GHGs",
    "mean_land": "Land Use",
    "mean_watscar": "Water Scarcity",
    "mean_bio": "Biodiversity",
    "mean_watuse": "Water Use"
}

# Build dimensions for parallel coordinates plot
dims = []
for col in envir_vars:
    dims.append(dict(
        label=labels_cha.get(col, col),
        values=da[col]
    ))
print("\nDimension information of parallel coordinates plot:")
for d in dims:
    print(d)

# plotly visualization
figure = go.Figure(data=go.Parcoords(
    line=dict(
        color=da['diet_code'],
        colorscale='Turbo',  # Vibrant color scale
        showscale=True,
        cmin=0,
        cmax=len(diet_tags) - 1
    ),
    dimensions=dims
))
figure.update_layout(
    title="Environmental Impact by Diet Type",
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(size=12),
)
figure.show()

# scatter matrix: more analytical view
# Focus on key features to highlight diet and water-related indicators
sns.set_theme(style="whitegrid")
color_palette = sns.color_palette("Set2", n_colors=len(diet_tags))

sns.pairplot(
    da,
    vars=["mean_watscar", "mean_watuse", "mean_ghgs"],
    hue="diet_group",
    palette=color_palette,
    diag_kind="kde"
)
plt.suptitle("Dietary Patterns & Key Environmental Indicators", y=1.02)
plt.tight_layout()
plt.show()
