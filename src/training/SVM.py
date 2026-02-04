from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2] 
    RESULTS_DIR = PROJECT_ROOT / "results"
    FEATURES_DIR = RESULTS_DIR / "features"
    MODELS_DIR = RESULTS_DIR / "models"
    PLOTS_DIR = RESULTS_DIR / "plots"

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    features_path = FEATURES_DIR / "mri_features_custom.npz"
    if not features_path.exists():
        raise SystemExit(
            f"File feature non trovato: {features_path}\n"
            "Esegui prima: python src/main.py --task extract_custom"
        )

    data = np.load(features_path, allow_pickle=True)
    X_train = data["train_feats"]
    y_train = data["train_labels"]
    X_test = data["val_feats"]
    y_test = data["val_labels"]

    if "classes" in data:
        class_names = data["classes"].tolist()
    else:
        class_names = np.unique(y_train).astype(str).tolist()

    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    
    param_grid = {
        "C": [500, 1000, 2000, 5000],
        "gamma": [0.008, 0.01, 0.012, 0.015],
        "kernel": ["rbf", "poly"],
    }

    grid = GridSearchCV(
        SVC(random_state=42, class_weight="balanced"),
        param_grid,
        refit=True,
        verbose=2,
        cv=5,
        n_jobs=-1,
    )

    grid.fit(X_train_s, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_s)
    test_acc = best_model.score(X_test_s, y_test)

    print(f"I migliori parametri trovati: {grid.best_params_}")
    print(f"Miglior accuratezza CV: {grid.best_score_:.4f}")
    print(f"Accuracy sul validation/test set: {test_acc:.4f}\n")

    print("Report Classificazione:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    
    svm_path = MODELS_DIR / "svm_best.joblib"
    scaler_path = MODELS_DIR / "svm_scaler.joblib"

    joblib.dump(best_model, svm_path)
    joblib.dump(scaler, scaler_path)

    
    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train_s)

    pca_path = MODELS_DIR / "svm_pca2d.joblib"
    joblib.dump(pca, pca_path)
    
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
    
    scatter = ax.scatter(
        X_train_pca[:, 0],
        X_train_pca[:, 1],
        c=y_train,
        s=20,
        alpha=0.8,
        edgecolors="k",
    )
    ax.set_title(f"SVM (PCA 2D) - Test Acc: {test_acc:.4f}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    handles, _ = scatter.legend_elements()
    ax.legend(handles, class_names, title="Classes", loc="upper left")

    plot_path = PLOTS_DIR / "svm_pca_scatter.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


    meta = {
        "class_names": class_names,
        "best_params": grid.best_params_,
        "cv_best_score": float(grid.best_score_),
        "test_acc": float(test_acc),
    }
    meta_path = MODELS_DIR / "svm_meta.json"
    import json
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nSalvati:")
    print(f"- SVM: {svm_path}")
    print(f"- Scaler: {scaler_path}")
    print(f"- PCA2D: {pca_path}")
    print(f"- Plot: {plot_path}")
    print(f"- Meta: {meta_path}")

if __name__ == "__main__":
    main()
