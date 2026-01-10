"""
Quick visualization of reserve regions in generated block model.
"""
import pandas as pd
import plotly.express as px

# Load the latest generated block model
csv_path = "data/block_model_10_10_10_2026_01_10_04_25.csv"
df = pd.read_csv(csv_path)

# Map reserve codes to labels
df['reserve_label'] = df['reserve'].map({
    1: 'Ore',
    0: 'Waste',
    -1: 'Overburden/Other'
})

# Create 3D scatter plot colored by reserve type
fig = px.scatter_3d(
    df,
    x='x',
    y='y',
    z='z',
    color='reserve_label',
    color_discrete_map={
        'Ore': 'gold',
        'Waste': 'red',
        'Overburden/Other': 'lightgray'
    },
    title='Block Model Reserve Regions',
    labels={'reserve_label': 'Reserve Type'},
    hover_data=['grade', 'economic_value']
)

fig.update_traces(marker=dict(size=5, opacity=0.8))
fig.update_layout(height=700)

fig.write_html("reserve_visualization.html")
print("Visualization saved to reserve_visualization.html")

# Print statistics
print("\n=== Reserve Statistics ===")
print(f"Total blocks: {len(df)}")
print(f"Ore blocks: {(df['reserve'] == 1).sum()}")
print(f"Waste blocks: {(df['reserve'] == 0).sum()}")
print(f"Overburden blocks: {(df['reserve'] == -1).sum()}")

print("\n=== Grade Statistics by Reserve Type ===")
print(df.groupby('reserve_label')['grade'].describe())

print("\n=== Economic Value Statistics by Reserve Type ===")
print(df.groupby('reserve_label')['economic_value'].describe())
