import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris=sns.load_dataset("iris")
iris.drop_duplicates(inplace=True)
# Z-score for petal length
col1=iris['petal_length'].dropna()
avg1=np.mean(col1)
print(f"Mean for petal length: {avg1}")
std1=np.std(col1)
print(f"Standard deviation for petal length: {std1}")
z1=np.abs((col1-avg1)/std1)

# Z-score for petal width
col2=iris['petal_width'].dropna()
avg2=np.mean(col2)
print(f"Mean for petal width: {avg2}")
std2=np.std(col2)
print(f"Standard deviation for petal width: {std2}")
z2=np.abs((col2-avg2)/std2)

# Z-score for sepal length
col3=iris['sepal_length'].dropna()
avg3=np.mean(col3)
print(f"Mean for sepal length: {avg3}")
std3=np.std(col3)
print(f"Standard deviation for sepal length: {std3}")
z3=np.abs((col3-avg3)/std3)

z_scores=pd.DataFrame({
    "petal_length": z1,
    "petal_width": z2,
    "sepal_length": z3
})
print(z_scores)

# Aggregated z-scores and outliers
aggregated_z=z_scores.mean(axis=1)
print(aggregated_z)
outliers=aggregated_z[aggregated_z>3]
print(outliers)

# Since there are no outliers, I am picking 10 elements with the largest z-scores as the outliers
outliers=aggregated_z.nlargest(20)
print(outliers)
iris['is_outlier']=iris.index.isin(outliers.index)
print(iris[iris['is_outlier']==True])

# Scatterplots
sns.scatterplot(
    data=iris,
    x="petal_length",
    y="petal_width",
    hue="is_outlier"
)
plt.title("Sepal length vs petal width")
plt.show()
sns.scatterplot(
    data=iris,
    x="sepal_length",
    y="petal_width",
    hue="is_outlier"
)
plt.title("Petal length vs petal width")
plt.show()
sns.scatterplot(
    data=iris,
    x="petal_length",
    y="sepal_length",
    hue="is_outlier"
)
plt.title("Petal length vs sepal length")
plt.show()
