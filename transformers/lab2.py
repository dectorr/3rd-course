import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

penguins=sns.load_dataset("penguins")
penguins.dropna(inplace=True)
print(penguins)
X=penguins.drop(columns=["species"])
y=penguins["species"]
X_train, X_temp, y_train, y_temp=train_test_split(
    X,y,
    test_size=0.30,
    random_state=42,
    stratify=y
)
X_val, X_test, y_val, y_test=train_test_split(
    X_temp,y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

numeric_cols=X_train.select_dtypes(include="number").columns
categorical_cols = X_train.select_dtypes(exclude="number").columns
train_mean=X_train[numeric_cols].mean()
train_std=X_train[numeric_cols].std()
X_train.loc[:,numeric_cols]=(X_train[numeric_cols]-train_mean)/train_std
X_val.loc[:, numeric_cols]=(X_val[numeric_cols]-train_mean)/train_std
X_test.loc[:, numeric_cols]=(X_test[numeric_cols]-train_mean)/train_std
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)
X_train_cat = pd.DataFrame(
    encoder.fit_transform(X_train[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols),
    index=X_train.index
)
X_val_cat = pd.DataFrame(
    encoder.transform(X_val[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols),
    index=X_val.index
)
X_test_cat = pd.DataFrame(
    encoder.transform(X_test[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols),
    index=X_test.index
)
X_train = pd.concat([X_train[numeric_cols], X_train_cat], axis=1)
X_val = pd.concat([X_val[numeric_cols], X_val_cat], axis=1)

X_test = pd.concat([X_test[numeric_cols], X_test_cat], axis=1)
print(X_train.head())
print(X_train.shape, X_val.shape, X_test.shape)

depths = [1,3,5]

for depth in depths:
    model=DecisionTreeClassifier(
        max_depth=depth,
        criterion="gini",
        random_state=42
    )
    model.fit(X_train,y_train)
    y_train_pred=model.predict(X_train)
    y_val_pred=model.predict(X_val)
    print(f"Depth:{depth}")
    print(f"Train accuracy: {accuracy_score(y_train,y_train_pred)}")
    print(f"Validation accuracy: {accuracy_score(y_val,y_val_pred)}")
    plt.figure(figsize=(24, 12))
    plot_tree(
        model,
        feature_names=X_train.columns,
        class_names=model.classes_,
        filled=True,
        rounded=True,
        fontsize=8
    )
    plt.title(f"Decision Tree (max_depth={depth})")
    plt.show()

np.random.seed(42)
noise=np.random.uniform(
    low=-0.1,
    high=0.1,
    size=X_train.shape
)
X_train_noisy = pd.DataFrame(
    X_train.to_numpy() + noise,
    columns=X_train.columns,
    index=X_train.index
)
for depth in depths:
    model = DecisionTreeClassifier(
        max_depth=depth,
        criterion="gini",
        random_state=42
    )
    model.fit(X_train_noisy,y_train)
    y_train_pred=model.predict(X_train_noisy)
    y_val_pred=model.predict(X_val)
    print(f"Depth:{depth}")
    print(f"Train accuracy: {accuracy_score(y_train, y_train_pred)}")
    print(f"Validation accuracy: {accuracy_score(y_val, y_val_pred)}")
    plt.figure(figsize=(24, 12))
    plot_tree(
        model,
        feature_names=X_train.columns,
        class_names=model.classes_,
        filled=True,
        rounded=True,
        fontsize=8
    )
    plt.title(f"Decision Tree, max_depth={depth}")
    plt.show()
pca_original=PCA()
X_train_pca_original=pca_original.fit_transform(X_train)
pca_noisy=PCA()
X_train_pca_noisy=pca_noisy.fit_transform(X_train_noisy)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, components, title in [
    (axes[0], X_train_pca_original, "PCA: original"),
    (axes[1], X_train_pca_noisy, "PCA: noisy")
]:
    for species in y_train.unique():
        mask = (y_train.to_numpy() == species)
        ax.scatter(
            components[mask, 0],
            components[mask, 1],
            label=species,
            alpha=0.7
        )
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(9, 5))
plt.plot(
    np.cumsum(pca_original.explained_variance_ratio_),
    marker="o",
    label="Original data"
)
plt.plot(
    np.cumsum(pca_noisy.explained_variance_ratio_),
    marker="o",
    label="Noisy data"
)
plt.axhline(
    y=0.95,
    color="red",
    linestyle="--",
    label="95% variance"
)
plt.xlabel("Component amount")
plt.ylabel("Accumulated explained variance")
plt.title("Different PCA")
plt.legend()
plt.grid(True)
plt.show()

pca_denoise=PCA(n_components=0.95)
X_train_reduced=pca_denoise.fit_transform(X_train_noisy)
X_train_denoised=pd.DataFrame(
    pca_denoise.inverse_transform(X_train_reduced),
    columns=X_train.columns,
    index=X_train.index
)
print("Components before PCA:", X_train.shape[1])
print("Components after PCA:", pca_denoise.n_components_)
print(
    "Explained variance:",
    pca_denoise.explained_variance_ratio_.sum()
)
print("\nDenoised:")
print(X_train_denoised.head())
fig, axes = plt.subplots(1,3,figsize=(18, 6))

datasets = [
    (X_train, "Original data"),
    (X_train_noisy, "Noisy data"),
    (X_train_denoised, "Denoised data")
]
for ax, (data, title) in zip(axes, datasets):
    im = ax.imshow(
        data.iloc[:30].to_numpy(),
        aspect="auto",
        cmap="viridis"
    )
    ax.set_title(title)
    ax.set_xlabel("Features")
    ax.set_ylabel("Examples")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
mse_noisy=np.mean((X_train.to_numpy()-X_train_noisy.to_numpy())**2)
mse_denoised=np.mean((X_train.to_numpy()-X_train_denoised)**2)
print("MSE noisy:", mse_noisy)
print("MSE denoised:", mse_denoised)
X_val_denoised=pd.DataFrame(
    pca_denoise.inverse_transform(
        pca_denoise.transform(X_val)
    ),
    columns=X_val.columns,
    index=X_val.index
)

results=[]
models={}

for depth in depths:
    for data_name, X_tr, X_va in [
        ("Noisy", X_train_noisy, X_val),
        ("Denoised", X_train_denoised, X_val)
    ]:
        model=DecisionTreeClassifier(
            max_depth=depth,
            criterion="gini",
            random_state=42
        )
        model.fit(X_tr, y_train)
        models[(data_name, depth)] = model
        results.append({
            "Data type": data_name,
            "Depth": depth,
            "Train accuracy": accuracy_score(y_train, model.predict(X_tr)),
            "Validation accuracy": accuracy_score(y_val, model.predict(X_va))
        })
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
for (data_name, depth), model in models.items():
    plt.figure(figsize=(20, 9))
    plot_tree(
        model,
        feature_names=X_train.columns,
        class_names=model.classes_,
        filled=True,
        rounded=True,
        fontsize=8
    )
    plt.title(f"{data_name}, max_depth={depth}")
    plt.show()
