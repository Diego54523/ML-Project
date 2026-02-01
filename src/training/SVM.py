from sklearn.preprocessing import StandardScaler  
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ML-Project/
    FEATURES_DIR = PROJECT_ROOT / "results" / "features"
    LOGS_DIR = PROJECT_ROOT / "results" / "logs"
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    features_path = FEATURES_DIR / "mri_features_custom.npz"
    if not features_path.exists():
        raise SystemExit(
            f"File feature non trovato: {features_path}\n"
            "Esegui prima l'estrazione delle features (Extract_Features_Custom_CNN.py)."
        )

    data = np.load(features_path, allow_pickle=True)
    X_train = data["train_feats"]
    y_train = data["train_labels"]
    X_test = data["val_feats"]
    y_test = data["val_labels"]

    if "classes" in data:
        class_names = data["classes"].tolist()
    else:
        class_names = np.unique(y_train).astype(str)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    param_grid = {
        'C': [500, 1000, 2000, 5000],          
        'gamma': [0.008, 0.01, 0.012, 0.015], 
        'kernel': ['rbf', 'poly']     
    }

    grid = GridSearchCV(
        SVC(random_state=42, class_weight='balanced'),
        param_grid,
        refit=True, 
        verbose=2,
        cv=5,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print(f"I migliori parametri trovati: {grid.best_params_}")
    print(f"Miglior accuratezza in validazione incrociata: {grid.best_score_:.4f}")

    best_model = grid.best_estimator_ 
    grid_predictions = best_model.predict(X_test)
    test_acc = best_model.score(X_test, y_test)

    print("\nReport Classificazione Dettagliato:")
    print(classification_report(y_test, grid_predictions, target_names=class_names))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, grid_predictions))
    print(f"Accuracy on test set: {test_acc:.4f}")

    pca = PCA(n_components=2) 
    X_train_pca = pca.fit_transform(X_train)

    viz_model = SVC(
        kernel=grid.best_params_['kernel'], 
        C=grid.best_params_['C'],
        gamma=grid.best_params_['gamma'],
        class_weight='balanced',
        random_state=42
    )
    viz_model.fit(X_train_pca, y_train)

    fig, ax = plt.subplots(figsize=(10, 8))

    DecisionBoundaryDisplay.from_estimator(
        viz_model,
        X_train_pca,
        response_method="predict",
        cmap=plt.cm.RdYlBu,
        alpha=0.6,
        ax=ax,
        grid_resolution=300,
        xlabel="Principal Component 1",
        ylabel="Principal Component 2",
    )

    scatter = plt.scatter(
        X_train_pca[:, 0],
        X_train_pca[:, 1],
        c=y_train,
        edgecolors="k",
        cmap=plt.cm.RdBu,
        s=20,
        alpha=0.8
    )

    ax.set_aspect('equal', adjustable='box')


    x_min, x_max = X_train_pca[:, 0].min(), X_train_pca[:, 0].max()
    y_min, y_max = X_train_pca[:, 1].min(), X_train_pca[:, 1].max()

    margin = 1.0 

    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    handles, _ = scatter.legend_elements()

    ax.legend(handles, class_names, title="Classes", loc="upper left", fontsize='small', title_fontsize='medium')
    
    ax.set_title(f"SVM Decision Boundary (2D Projection)\nTest Accuracy: {test_acc:.4f}")
    
    plt.show()

if __name__ == "__main__":
    main()

