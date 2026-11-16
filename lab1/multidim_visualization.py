"""Визуализация многомерных данных и кластеризация: HighDimVisualizer и ClusteringVisualizer."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import linear_sum_assignment
from sklearn import datasets as sk_datasets
from sklearn.datasets import make_classification, make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import f_classif
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             calinski_harabasz_score, davies_bouldin_score,
                             adjusted_rand_score, adjusted_mutual_info_score,
                             homogeneity_completeness_v_measure)
from sklearn.metrics.cluster import contingency_matrix
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')

try:
    from IPython.display import display
except ImportError:  # запуск вне Jupyter
    display = print

try:
    import umap  # pip install umap-learn
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# Настройки отображения
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100


class HighDimVisualizer:
    """Класс для визуализации многомерных данных."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.class_names = None
        np.random.seed(random_state)

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------
    def _class_label(self, c):
        """Человекочитаемое имя класса (если имена известны)."""
        if self.class_names is not None and 0 <= int(c) < len(self.class_names):
            return str(self.class_names[int(c)])
        return f'Class {int(c)}'

    def _add_class_legend(self, ax, y, cmap_name, title="Классы", max_items=10):
        """Легенда «цвет -> класс» для scatter, раскрашенного числовыми метками."""
        classes = np.unique(y)
        cmap = plt.get_cmap(cmap_name)
        norm = plt.Normalize(vmin=classes.min(), vmax=classes.max())
        handles = [Line2D([0], [0], marker='o', linestyle='', markersize=7,
                          markerfacecolor=cmap(norm(c)), markeredgecolor='k',
                          label=self._class_label(c))
                   for c in classes[:max_items]]
        ax.legend(handles=handles, title=title, fontsize=8)

    # ------------------------------------------------------------------
    # Генерация синтетических данных
    # ------------------------------------------------------------------
    def generate_sample_data(self, n_samples=1000, n_features=10, n_classes=3):
        """Генерация тестовых данных."""
        print(f"Генерация данных: {n_samples} образцов, {n_features} признаков, {n_classes} классов")

        # Генерируем данные с классами
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.8),
            n_redundant=int(n_features * 0.2),
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=self.random_state
        )

        # Добавляем немного шума
        X += np.random.normal(0, 0.1, X.shape)

        # Создаем имена признаков
        feature_names = [f'Feature_{i+1}' for i in range(n_features)]

        # Масштабируем данные
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y, feature_names

    def generate_complex_data(self, n_samples=500):
        """Генерация более сложных данных для демонстрации."""
        print("Генерация комплексных данных...")

        # Создаем несколько наборов данных
        X1, y1 = make_blobs(n_samples=n_samples, centers=4, n_features=10,
                            cluster_std=1.5, random_state=self.random_state)

        X2, y2 = make_moons(n_samples=n_samples, noise=0.1, random_state=self.random_state)
        # Полумесяцы двумерные: дополняем их нулями до 10 признаков, чтобы
        # можно было объединить с blobs
        X2 = np.hstack([X2 + [5, 0], np.zeros((n_samples, X1.shape[1] - 2))])

        # Объединяем данные
        X = np.vstack([X1, X2])
        y = np.hstack([y1, y2 + 4])

        # Добавляем дополнительные случайные признаки
        n_extra_features = 8
        extra_features = np.random.randn(X.shape[0], n_extra_features)
        X = np.hstack([X, extra_features])

        # Масштабируем
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        feature_names = [f'F{i+1}' for i in range(X.shape[1])]

        return X, y, feature_names

    # ------------------------------------------------------------------
    # Визуализации
    # ------------------------------------------------------------------
    def plot_pairplot(self, X, y, feature_names, n_features=5, indices=None):
        """Парные диаграммы рассеяния для n признаков."""
        print("\n1. Парные диаграммы рассеяния (PairPlot)")

        n_features = min(n_features, X.shape[1])
        if indices is None:
            indices = np.random.choice(X.shape[1], n_features, replace=False)

        df = pd.DataFrame(X[:, indices], columns=[feature_names[i] for i in indices])
        df['Class'] = [self._class_label(c) for c in y]

        g = sns.pairplot(df, hue='Class', palette='viridis',
                         diag_kind='kde', plot_kws={'alpha': 0.6})
        g.fig.suptitle(f'Pairplot для {n_features} признаков', y=1.02)
        plt.show()

    def plot_correlation_heatmap(self, X, feature_names):
        """Тепловая карта корреляций между признаками."""
        print("\n2. Тепловая карта корреляций")

        df = pd.DataFrame(X, columns=feature_names)
        corr_matrix = df.corr()

        size = max(8, 0.6 * len(feature_names))
        plt.figure(figsize=(size + 2, size))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                    annot=len(feature_names) <= 20, fmt=".2f")
        plt.title('Матрица корреляций признаков', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_pca_2d_3d(self, X, y):
        """Визуализация PCA в 2D и 3D."""
        print("\n3. Визуализация PCA (2D и 3D)")

        pca = PCA(n_components=3, random_state=self.random_state)
        X_pca = pca.fit_transform(X)

        explained_variance = pca.explained_variance_ratio_
        print(f"Объясненная дисперсия: {explained_variance}")
        print(f"Суммарная объясненная дисперсия: {sum(explained_variance):.3f}")

        fig = plt.figure(figsize=(15, 6))

        # 2D plot
        ax2d = fig.add_subplot(121)
        ax2d.scatter(X_pca[:, 0], X_pca[:, 1], c=y,
                     cmap='viridis', alpha=0.7, edgecolors='k', linewidth=0.5)
        ax2d.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
        ax2d.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
        ax2d.set_title('PCA 2D проекция')
        ax2d.grid(True, alpha=0.3)
        self._add_class_legend(ax2d, y, 'viridis')

        # 3D plot
        ax3d = fig.add_subplot(122, projection='3d')
        ax3d.scatter3D(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                       c=y, cmap='viridis', alpha=0.7, edgecolors='k', linewidth=0.5)
        ax3d.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
        ax3d.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
        ax3d.set_zlabel(f'PC3 ({explained_variance[2]:.2%})')
        ax3d.set_title('PCA 3D проекция')

        plt.tight_layout()
        plt.show()

        self.plot_explained_variance(X)

    def plot_explained_variance(self, X):
        """График объясненной дисперсии PCA (по всем главным компонентам)."""
        explained_variance = PCA().fit(X).explained_variance_ratio_

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
        plt.xlabel('Номер главной компоненты')
        plt.ylabel('Доля объясненной дисперсии')
        plt.title('Объясненная дисперсия по компонентам')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        cumulative_variance = np.cumsum(explained_variance)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                 'b-o', linewidth=2, markersize=6)
        plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% дисперсии')
        plt.axhline(y=0.90, color='g', linestyle='--', alpha=0.7, label='90% дисперсии')
        plt.xlabel('Количество компонент')
        plt.ylabel('Кумулятивная объясненная дисперсия')
        plt.title('Кумулятивная объясненная дисперсия')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.show()

        for thr in (0.80, 0.90, 0.95):
            k = int(np.searchsorted(cumulative_variance, thr) + 1)
            print(f"  Для {thr:.0%} дисперсии нужно компонент: {k}")

    def plot_tsne_visualization(self, X, y, perplexities=(5, 30, 50)):
        """Визуализация t-SNE проекции."""
        print("\n4. Визуализация t-SNE")

        # Для больших датасетов можно использовать PCA для предобработки
        if X.shape[0] > 1000 and X.shape[1] > 50:
            print("  Применяем PCA для уменьшения размерности перед t-SNE...")
            X_reduced = PCA(n_components=50, random_state=self.random_state).fit_transform(X)
        else:
            X_reduced = X

        # perplexity должна быть меньше числа объектов
        perplexities = [p for p in perplexities if p < X.shape[0]]
        fig, axes = plt.subplots(1, len(perplexities), figsize=(6 * len(perplexities), 5))
        axes = np.atleast_1d(axes)

        for i, perplexity in enumerate(perplexities):
            tsne = TSNE(n_components=2, perplexity=perplexity,
                        random_state=self.random_state, max_iter=1000)
            X_tsne = tsne.fit_transform(X_reduced)

            axes[i].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y,
                            cmap='tab10', alpha=0.7, s=30, edgecolors='k', linewidth=0.3)
            axes[i].set_title(f't-SNE (perplexity={perplexity})')
            axes[i].set_xlabel('t-SNE 1')
            axes[i].set_ylabel('t-SNE 2')
            axes[i].grid(True, alpha=0.3)
            if i == 0:
                self._add_class_legend(axes[i], y, 'tab10')

        plt.tight_layout()
        plt.show()

    def plot_umap_visualization(self, X, y):
        """Визуализация UMAP проекции."""
        print("\n5. Визуализация UMAP")

        if not HAS_UMAP:
            print("UMAP не установлен. Установите: pip install umap-learn")
            return

        n_neighbors_list = [5, 15, 50]
        min_dists = [0.1, 0.5, 0.99]

        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        axes = axes.flatten()

        plot_idx = 0
        for n_neighbors in n_neighbors_list:
            for min_dist in min_dists:
                umap_model = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                                       min_dist=min_dist, random_state=self.random_state)
                X_umap = umap_model.fit_transform(X)

                axes[plot_idx].scatter(X_umap[:, 0], X_umap[:, 1], c=y,
                                       cmap='Spectral', alpha=0.7, s=30,
                                       edgecolors='k', linewidth=0.3)
                axes[plot_idx].set_title(f'UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})')
                axes[plot_idx].set_xlabel('UMAP 1')
                axes[plot_idx].set_ylabel('UMAP 2')
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1

        plt.tight_layout()
        plt.show()

    def plot_parallel_coordinates(self, X, y, feature_names, n_features=10):
        """Параллельные координаты."""
        print("\n6. Параллельные координаты")

        n_features = min(n_features, X.shape[1])
        indices = np.random.choice(X.shape[1], n_features, replace=False)

        df = pd.DataFrame(X[:, indices], columns=[feature_names[i] for i in indices])
        df_scaled = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
        df_scaled['Class'] = y

        plt.figure(figsize=(14, 8))

        # Выбираем подмножество точек для избежания перегруженности
        n_samples_plot = min(200, len(df_scaled))
        sample_indices = np.random.choice(len(df_scaled), n_samples_plot, replace=False)
        df_plot = df_scaled.iloc[sample_indices]

        unique_classes = np.unique(y)
        colors_plt = cm.tab10(np.linspace(0, 1, len(unique_classes)))

        for i, class_val in enumerate(unique_classes):
            class_data = df_plot[df_plot['Class'] == class_val].iloc[:, :-1]
            for j in range(len(class_data)):
                plt.plot(range(n_features), class_data.iloc[j],
                         color=colors_plt[i], alpha=0.3, linewidth=0.8)

        plt.xticks(range(n_features), [feature_names[i] for i in indices], rotation=45)
        plt.xlabel('Признаки')
        plt.ylabel('Нормализованное значение')
        plt.title('Параллельные координаты (случайная выборка)')
        plt.grid(True, alpha=0.3)

        legend_elements = [Line2D([0], [0], color=colors_plt[i], lw=2,
                                  label=self._class_label(class_val))
                           for i, class_val in enumerate(unique_classes)]
        plt.legend(handles=legend_elements, title="Классы")

        plt.tight_layout()
        plt.show()

    def plot_radar_chart(self, X, y, feature_names, n_features=6):
        """Радар-чарты для центроидов классов."""
        print("\n7. Радар-чарты по классам")

        n_features = min(n_features, X.shape[1])
        indices = np.random.choice(X.shape[1], n_features, replace=False)
        selected_features = [feature_names[i] for i in indices]

        unique_classes = np.unique(y)
        fig, axes = plt.subplots(1, len(unique_classes), figsize=(5 * len(unique_classes), 5),
                                 subplot_kw=dict(projection='polar'))
        axes = np.atleast_1d(axes)

        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
        angles += angles[:1]  # Замыкаем круг

        for idx, class_val in enumerate(unique_classes):
            class_mean = X[y == class_val][:, indices].mean(axis=0)
            values = class_mean.tolist()
            values += values[:1]

            axes[idx].plot(angles, values, 'o-', linewidth=2)
            axes[idx].fill(angles, values, alpha=0.25)
            axes[idx].set_xticks(angles[:-1])
            axes[idx].set_xticklabels(selected_features, fontsize=8)
            axes[idx].set_title(self._class_label(class_val), size=14, y=1.1)
            axes[idx].grid(True)

        plt.suptitle('Радар-чарты средних значений признаков по классам', fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_interactive_3d(self, X, y, feature_names=None, filename="interactive_3d_plot.html"):
        """Интерактивная 3D визуализация с использованием Plotly."""
        print("\n8. Интерактивная 3D визуализация")

        pca = PCA(n_components=3, random_state=self.random_state)
        X_pca = pca.fit_transform(X)

        fig = go.Figure(data=[
            go.Scatter3d(
                x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
                mode='markers',
                marker=dict(size=5, color=y, colorscale='Viridis', opacity=0.8, showscale=True),
                text=[f'Class: {self._class_label(label)}<br>Point: {i}' for i, label in enumerate(y)],
                hoverinfo='text'
            )
        ])

        explained_variance = pca.explained_variance_ratio_
        fig.update_layout(
            title=f'3D PCA проекция данных<br>Объясненная дисперсия: {sum(explained_variance):.2%}',
            scene=dict(
                xaxis_title=f'PC1 ({explained_variance[0]:.2%})',
                yaxis_title=f'PC2 ({explained_variance[1]:.2%})',
                zaxis_title=f'PC3 ({explained_variance[2]:.2%})'
            ),
            width=1000, height=800
        )

        if filename:
            fig.write_html(filename)
            print(f"  График сохранен в файл: {filename}")
        fig.show()

    def compare_dim_reduction_methods(self, X, y):
        """Сравнение различных методов уменьшения размерности."""
        print("\n9. Сравнение методов уменьшения размерности")

        methods = [
            ('PCA', PCA(n_components=2, random_state=self.random_state)),
            ('t-SNE', TSNE(n_components=2, random_state=self.random_state,
                           perplexity=min(30, (X.shape[0] - 1) / 3), max_iter=1000)),
            ('MDS', MDS(n_components=2, random_state=self.random_state, n_init=1, max_iter=300)),
            ('Isomap', Isomap(n_components=2, n_neighbors=10)),
        ]
        if HAS_UMAP:
            methods.append(('UMAP', umap.UMAP(n_components=2, random_state=self.random_state)))

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (name, model) in enumerate(methods):
            print(f"  Применяем {name}...")
            try:
                if name == 't-SNE' and X.shape[0] > 1000 and X.shape[1] > 50:
                    X_in = PCA(n_components=50, random_state=self.random_state).fit_transform(X)
                else:
                    X_in = X
                X_emb = model.fit_transform(X_in)

                axes[idx].scatter(X_emb[:, 0], X_emb[:, 1], c=y,
                                  cmap='tab10', alpha=0.7, s=30,
                                  edgecolors='k', linewidth=0.3)
                axes[idx].set_title(f'{name}')
                axes[idx].set_xlabel('Component 1')
                axes[idx].set_ylabel('Component 2')
                axes[idx].grid(True, alpha=0.3)
                if idx == 0:
                    self._add_class_legend(axes[idx], y, 'tab10')
            except Exception as e:
                axes[idx].text(0.5, 0.5, f'Error:\n{str(e)}',
                               ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'{name} (failed)')

        for idx in range(len(methods), len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle('Сравнение методов уменьшения размерности', fontsize=16)
        plt.tight_layout()
        plt.show()

    def run_full_visualization(self, X, y, feature_names):
        """Запуск полной визуализации."""
        print("=" * 60)
        print("ПОЛНАЯ ВИЗУАЛИЗАЦИЯ МНОГОМЕРНЫХ ДАННЫХ")
        print("=" * 60)

        self.plot_pairplot(X, y, feature_names, n_features=5)
        self.plot_correlation_heatmap(X, feature_names)
        self.plot_pca_2d_3d(X, y)
        self.plot_tsne_visualization(X, y)
        self.plot_parallel_coordinates(X, y, feature_names, n_features=8)
        self.plot_radar_chart(X, y, feature_names, n_features=6)
        self.compare_dim_reduction_methods(X, y)
        self.plot_umap_visualization(X, y)
        self.plot_interactive_3d(X, y, feature_names)

        print("\n" + "=" * 60)
        print("ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА")
        print("=" * 60)


class ClusteringVisualizer(HighDimVisualizer):
    """Кластеризация и визуализация произвольного табличного датасета."""

    ALGORITHMS = ('KMeans', 'Agglomerative', 'GMM', 'DBSCAN')
    SCALERS = {'standard': StandardScaler, 'minmax': MinMaxScaler, 'robust': RobustScaler}
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def __init__(self, data, target=None, feature_columns=None, class_names=None,
                 scaling='standard', dataset_name='Dataset', random_state=42):
        super().__init__(random_state=random_state)
        self.dataset_name = dataset_name
        self.scaling = scaling

        df, self.y, self.class_names = self._split_target(data, target, class_names)
        self.X_df = self._prepare_features(df, feature_columns)
        self.feature_names = list(self.X_df.columns)
        self.X = self._scale(self.X_df.values.astype(float))

        self.n_clusters = None
        self.labels_ = {}
        self.models_ = {}
        self.embeddings_ = {}
        self.reducers_ = {}
        self.metrics_ = None
        self.k_search_ = None

    # ------------------------------------------------------------------
    # Загрузка данных
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(cls, path, target=None, **kwargs):
        """Создание визуализатора из CSV-файла (например, датасета с Kaggle)."""
        df = pd.read_csv(path)
        kwargs.setdefault('dataset_name', str(path).split('/')[-1])
        return cls(df, target=target, **kwargs)

    @classmethod
    def from_sklearn(cls, name='wine', **kwargs):
        """Создание визуализатора из встроенного датасета sklearn: wine, iris, breast_cancer."""
        loaders = {'wine': sk_datasets.load_wine, 'iris': sk_datasets.load_iris,
                   'breast_cancer': sk_datasets.load_breast_cancer}
        bunch = loaders[name]()
        df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        kwargs.setdefault('dataset_name', name.capitalize())
        kwargs.setdefault('class_names', list(bunch.target_names))
        return cls(df, target=bunch.target, **kwargs)

    @staticmethod
    def _split_target(data, target, class_names):
        """Отделяет вектор классов от признаков и кодирует классы числами 0..K-1."""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            data = np.asarray(data)
            df = pd.DataFrame(data, columns=[f'F{i+1}' for i in range(data.shape[1])])

        if target is None:
            return df, None, class_names

        if isinstance(target, str):
            y_raw = df[target]
            df = df.drop(columns=[target])
        else:
            y_raw = pd.Series(np.asarray(target))

        # Объекты без метки класса выбрасываем
        keep = ~pd.isna(y_raw).values
        df, y_raw = df.loc[keep].reset_index(drop=True), y_raw[keep].reset_index(drop=True)

        codes, uniques = pd.factorize(y_raw, sort=True)
        if class_names is None:
            class_names = [str(u) for u in uniques]
        return df, codes.astype(int), list(class_names)

    @staticmethod
    def _prepare_features(df, feature_columns, max_categories=20):
        """Предобработка признаков:"""
        if feature_columns is not None:
            df = df[list(feature_columns)].copy()
        df = df.dropna(axis=1, how='all')

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]

        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())

        dropped = []
        for c in cat_cols:
            if df[c].nunique() > max_categories:
                dropped.append(c)
                df = df.drop(columns=[c])
            else:
                df[c] = df[c].fillna(df[c].mode().iloc[0])
        cat_cols = [c for c in cat_cols if c not in dropped]
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, dtype=float)
        if dropped:
            print(f"Отброшены категориальные признаки с > {max_categories} значений: {dropped}")

        constant = [c for c in df.columns if df[c].nunique() <= 1]
        if constant:
            print(f"Отброшены константные признаки: {constant}")
            df = df.drop(columns=constant)
        return df.astype(float)

    def _scale(self, X):
        if self.scaling is None:
            self.scaler_ = None
            return X
        self.scaler_ = self.SCALERS[self.scaling]()
        return self.scaler_.fit_transform(X)

    # ------------------------------------------------------------------
    # Первичный анализ
    # ------------------------------------------------------------------
    def describe_data(self):
        """Краткое описание набора данных."""
        print(f"Набор данных: {self.dataset_name}")
        print(f"  Объектов: {self.X.shape[0]}, признаков после предобработки: {self.X.shape[1]}")
        print(f"  Масштабирование: {self.scaling}")
        if self.y is not None:
            counts = pd.Series(self.y).value_counts().sort_index()
            print(f"  Классов: {len(counts)}")
            for c, n in counts.items():
                print(f"    {self._class_label(c):>20}: {n} ({n / len(self.y):.1%})")
        display(self.X_df.describe().T.round(3))

    def top_features(self, n=5):
        """Наиболее «разделяющие» признаки по F-статистике дисперсионного анализа"""
        if self.y is None:
            order = np.argsort(self.X.var(axis=0))[::-1]
        else:
            f_values, _ = f_classif(self.X, self.y)
            order = np.argsort(np.nan_to_num(f_values))[::-1]
        return order[:n]

    def plot_eda(self, n_features=5):
        """Pairplot самых информативных признаков и тепловая карта корреляций."""
        idx = self.top_features(n_features)
        print("Самые информативные признаки (F-тест):", [self.feature_names[i] for i in idx])
        y = self.y if self.y is not None else np.zeros(len(self.X), dtype=int)
        self.plot_pairplot(self.X, y, self.feature_names, n_features=n_features, indices=idx)
        self.plot_correlation_heatmap(self.X, self.feature_names)

    def plot_scaling_effect(self):
        """Распределения признаков до и после масштабирования."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.boxplot(data=self.X_df, orient='h', ax=axes[0])
        axes[0].set_title('Исходные признаки (логарифмическая шкала)')
        axes[0].set_xscale('symlog', linthresh=1)
        axes[0].set_xlim(left=0)
        sns.boxplot(data=pd.DataFrame(self.X, columns=self.feature_names), orient='h', ax=axes[1])
        axes[1].set_title(f'После масштабирования ({self.scaling})')
        axes[1].set_yticklabels([])
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Подбор гиперпараметров
    # ------------------------------------------------------------------
    def find_optimal_k(self, k_range=range(2, 11)):
        """Подбор числа кластеров: метод «локтя» (inertia KMeans), силуэт,"""
        rows = []
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(self.X)
            gmm = GaussianMixture(n_components=k, random_state=self.random_state).fit(self.X)
            rows.append({
                'k': k,
                'Inertia': km.inertia_,
                'Silhouette': silhouette_score(self.X, km.labels_),
                'Calinski-Harabasz': calinski_harabasz_score(self.X, km.labels_),
                'Davies-Bouldin': davies_bouldin_score(self.X, km.labels_),
                'GMM BIC': gmm.bic(self.X),
            })
        res = pd.DataFrame(rows).set_index('k')
        self.k_search_ = res

        fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
        best = {'Inertia': None, 'Silhouette': res['Silhouette'].idxmax(),
                'Calinski-Harabasz': res['Calinski-Harabasz'].idxmax(),
                'Davies-Bouldin': res['Davies-Bouldin'].idxmin(),
                'GMM BIC': res['GMM BIC'].idxmin()}
        for ax, col in zip(axes, res.columns):
            ax.plot(res.index, res[col], 'o-', linewidth=2)
            if best[col] is not None:
                ax.axvline(best[col], color='r', linestyle='--', alpha=0.7, label=f'лучшее k={best[col]}')
            if self.y is not None:
                ax.axvline(len(np.unique(self.y)), color='g', linestyle=':', alpha=0.8, label='число классов')
            ax.set_title(col)
            ax.set_xlabel('k')
            ax.legend(fontsize=8)
        plt.suptitle(f'Подбор числа кластеров — {self.dataset_name}', fontsize=14)
        plt.tight_layout()
        plt.show()

        self.suggested_k_ = int(best['Silhouette'])
        print(f"Рекомендация по силуэту: k = {self.suggested_k_}")
        return res

    def estimate_dbscan_eps(self, min_samples=5, plot=True, X=None):
        """Оценка eps для DBSCAN по графику k-расстояний: сортируем расстояние"""
        X = self.X if X is None else X
        nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
        dist, _ = nn.kneighbors(X)
        d = np.sort(dist[:, -1])
        x = np.arange(len(d))
        p1, p2 = np.array([0, d[0]]), np.array([len(d) - 1, d[-1]])
        line = (p2 - p1) / np.linalg.norm(p2 - p1)
        vec = np.stack([x, d], axis=1) - p1
        dist_to_line = np.abs(vec[:, 0] * line[1] - vec[:, 1] * line[0])
        knee = int(np.argmax(dist_to_line))
        eps = float(d[knee])

        if plot:
            plt.figure(figsize=(8, 4))
            plt.plot(x, d, linewidth=2)
            plt.axhline(eps, color='r', linestyle='--', label=f'eps ≈ {eps:.2f}')
            plt.scatter([knee], [eps], color='r', zorder=3)
            plt.xlabel('Объекты, отсортированные по расстоянию')
            plt.ylabel(f'Расстояние до {min_samples}-го соседа')
            plt.title('График k-расстояний для выбора eps (DBSCAN)')
            plt.legend()
            plt.tight_layout()
            plt.show()
        return eps

    # ------------------------------------------------------------------
    # Кластеризация и оценка
    # ------------------------------------------------------------------
    def fit_clustering(self, n_clusters=None, algorithms=ALGORITHMS,
                       dbscan_eps=None, dbscan_min_samples=5):
        """Запускает выбранные алгоритмы кластеризации и сохраняет метки."""
        if n_clusters is None:
            if self.y is not None:
                n_clusters = len(np.unique(self.y))
            else:
                n_clusters = getattr(self, 'suggested_k_', 3)
        self.n_clusters = n_clusters

        for name in algorithms:
            if name == 'KMeans':
                model = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.random_state)
            elif name == 'Agglomerative':
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            elif name == 'GMM':
                model = GaussianMixture(n_components=n_clusters, covariance_type='full',
                                        n_init=5, random_state=self.random_state)
            elif name == 'DBSCAN':
                eps = dbscan_eps or self.estimate_dbscan_eps(dbscan_min_samples, plot=False)
                model = DBSCAN(eps=eps, min_samples=dbscan_min_samples)
            else:
                raise ValueError(f'Неизвестный алгоритм: {name}')
            self.labels_[name] = np.asarray(model.fit_predict(self.X))
            self.models_[name] = model
            n_found = len(set(self.labels_[name]) - {-1})
            print(f"{name:>14}: найдено кластеров = {n_found}")
        self.metrics_ = None
        return self.labels_

    def fit_dbscan_on_embedding(self, method='UMAP', n_components=2,
                                eps=None, min_samples=10, plot=True):
        """DBSCAN в пространстве пониженной размерности."""
        emb = self.get_embedding(method, n_components)
        if eps is None:
            eps = self.estimate_dbscan_eps(min_samples, plot=plot, X=emb)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        name = f'DBSCAN ({method})'
        self.add_labels(name, model.fit_predict(emb), model)
        n_found = len(set(self.labels_[name]) - {-1})
        print(f"{name}: eps={eps:.3f}, min_samples={min_samples}, найдено кластеров = {n_found}")
        return self.labels_[name]

    def add_labels(self, name, labels, model=None):
        """Добавляет внешнее разбиение (например, полученное вручную) к сравнению."""
        self.labels_[name] = np.asarray(labels)
        if model is not None:
            self.models_[name] = model
        self.metrics_ = None

    @staticmethod
    def cluster_accuracy(y_true, labels):
        """Доля правильно сгруппированных объектов при наилучшем однозначном"""
        C = contingency_matrix(y_true, labels)
        rows, cols = linear_sum_assignment(-C)
        return C[rows, cols].sum() / C.sum()

    def evaluate_clustering(self):
        """Внутренние метрики (по признаковому пространству, без шумовых точек)"""
        rows = []
        for name, labels in self.labels_.items():
            mask = labels != -1
            n_found = len(np.unique(labels[mask]))
            row = {'Algorithm': name, 'Clusters': n_found, 'Noise, %': 100 * (~mask).mean()}
            if n_found >= 2:
                row['Silhouette'] = silhouette_score(self.X[mask], labels[mask])
                row['Calinski-Harabasz'] = calinski_harabasz_score(self.X[mask], labels[mask])
                row['Davies-Bouldin'] = davies_bouldin_score(self.X[mask], labels[mask])
            if self.y is not None:
                h, c, v = homogeneity_completeness_v_measure(self.y, labels)
                row.update({'ARI': adjusted_rand_score(self.y, labels),
                            'AMI': adjusted_mutual_info_score(self.y, labels),
                            'Homogeneity': h, 'Completeness': c, 'V-measure': v,
                            'Accuracy': self.cluster_accuracy(self.y, labels)})
            rows.append(row)
        self.metrics_ = pd.DataFrame(rows).set_index('Algorithm')
        return self.metrics_

    def best_algorithm(self):
        """Лучший алгоритм: по ARI (если есть классы), иначе по силуэту."""
        self._ensure_metrics()
        col = 'ARI' if 'ARI' in self.metrics_ else 'Silhouette'
        return self.metrics_[col].idxmax()

    def _ensure_metrics(self):
        if self.metrics_ is None or set(self.metrics_.index) != set(self.labels_):
            self.evaluate_clustering()

    # ------------------------------------------------------------------
    # Понижение размерности для визуализации
    # ------------------------------------------------------------------
    def get_embedding(self, method='PCA', n_components=2):
        """Проекция данных в 2D/3D (результаты кэшируются)."""
        key = (method, n_components)
        if key in self.embeddings_:
            return self.embeddings_[key]
        if method == 'PCA':
            reducer = PCA(n_components=n_components, random_state=self.random_state)
        elif method == 't-SNE':
            reducer = TSNE(n_components=n_components, perplexity=min(30, (len(self.X) - 1) / 3),
                           init='pca', random_state=self.random_state, max_iter=1000)
        elif method == 'UMAP':
            if not HAS_UMAP:
                raise ImportError('UMAP не установлен: pip install umap-learn')
            reducer = umap.UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1,
                                random_state=self.random_state)
        elif method == 'Isomap':
            reducer = Isomap(n_components=n_components, n_neighbors=10)
        else:
            raise ValueError(f'Неизвестный метод: {method}')
        emb = reducer.fit_transform(self.X)
        self.embeddings_[key] = emb
        self.reducers_[key] = reducer
        return emb

    def _centroids_in_embedding(self, algorithm, method, n_components):
        """Центры KMeans/GMM, спроецированные в пространство PCA."""
        if method != 'PCA' or algorithm not in ('KMeans', 'GMM'):
            return None
        model = self.models_[algorithm]
        centers = model.cluster_centers_ if algorithm == 'KMeans' else model.means_
        return self.reducers_[(method, n_components)].transform(centers)

    def _scatter_labels(self, ax, emb, labels, title, is_true=False, three_d=False):
        """Рисует точки, раскрашенные по меткам; шум DBSCAN — серыми крестиками."""
        for i, lab in enumerate(np.unique(labels)):
            m = labels == lab
            if lab == -1 and not is_true:
                style = dict(c='lightgrey', marker='x', s=20, label='шум')
            else:
                name = self._class_label(lab) if is_true else f'кластер {lab}'
                style = dict(color=self.COLORS[int(lab) % len(self.COLORS)], s=35,
                             edgecolors='k', linewidths=0.3, alpha=0.85, label=name)
            coords = [emb[m, 0], emb[m, 1]] + ([emb[m, 2]] if three_d else [])
            ax.scatter(*coords, **style)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='best')

    def plot_clusters_2d(self, method='PCA', algorithms=None):
        """Истинные классы и результаты всех алгоритмов в одной 2D-проекции."""
        algorithms = list(algorithms or self.labels_.keys())
        self._ensure_metrics()
        emb = self.get_embedding(method, 2)
        n_panels = len(algorithms) + (self.y is not None)
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.8))
        axes = np.atleast_1d(axes)
        k = 0
        if self.y is not None:
            self._scatter_labels(axes[0], emb, self.y, 'Истинные классы', is_true=True)
            k = 1
        for ax, name in zip(axes[k:], algorithms):
            title = name
            if self.metrics_ is not None and 'ARI' in self.metrics_:
                title += f"\nARI={self.metrics_.loc[name, 'ARI']:.3f}"
            self._scatter_labels(ax, emb, self.labels_[name], title)
            centers = self._centroids_in_embedding(name, method, 2)
            if centers is not None:
                ax.scatter(centers[:, 0], centers[:, 1], marker='*', s=350,
                           c='black', edgecolors='white', label='центры')
        for ax in axes:
            ax.set_xlabel(f'{method} 1')
            ax.set_ylabel(f'{method} 2')
        plt.suptitle(f'Кластеризация в проекции {method} — {self.dataset_name}', fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_clusters_3d(self, method='PCA', algorithm=None):
        """Статичная 3D-визуализация: истинные классы и найденные кластеры."""
        algorithm = algorithm or self.best_algorithm()
        emb = self.get_embedding(method, 3)
        fig = plt.figure(figsize=(16, 7))
        panels = [(self.y, 'Истинные классы', True)] if self.y is not None else []
        panels.append((self.labels_[algorithm], f'Кластеры {algorithm}', False))
        for i, (labels, title, is_true) in enumerate(panels, 1):
            ax = fig.add_subplot(1, len(panels), i, projection='3d')
            self._scatter_labels(ax, emb, labels, title, is_true=is_true, three_d=True)
            if not is_true:
                centers = self._centroids_in_embedding(algorithm, method, 3)
                if centers is not None:
                    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='*',
                               s=400, c='black', label='центры')
            ax.set_xlabel(f'{method} 1')
            ax.set_ylabel(f'{method} 2')
            ax.set_zlabel(f'{method} 3')
            ax.view_init(elev=20, azim=45)
        if method == 'PCA':
            ev = self.reducers_[(method, 3)].explained_variance_ratio_
            fig.suptitle(f'3D-проекция PCA — {self.dataset_name} '
                         f'(объяснено {ev.sum():.1%} дисперсии)', fontsize=14)
        else:
            fig.suptitle(f'3D-проекция {method} — {self.dataset_name}', fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_interactive_3d_clusters(self, method='PCA', algorithm=None, filename=None):
        """Интерактивная 3D-визуализация (plotly): цвет - кластер, форма маркера —"""
        algorithm = algorithm or self.best_algorithm()
        emb = self.get_embedding(method, 3)
        labels = self.labels_[algorithm]
        symbols = ['circle', 'diamond', 'square', 'cross', 'x',
                   'circle-open', 'diamond-open', 'square-open']

        fig = go.Figure()
        for lab in np.unique(labels):
            m = labels == lab
            name = 'шум' if lab == -1 else f'кластер {lab}'
            sym = ([symbols[c % len(symbols)] for c in self.y[m]]
                   if self.y is not None else 'circle')
            hover = [f'#{j}<br>{name}' + (f'<br>класс: {self._class_label(self.y[j])}'
                                          if self.y is not None else '')
                     for j in np.where(m)[0]]
            fig.add_trace(go.Scatter3d(
                x=emb[m, 0], y=emb[m, 1], z=emb[m, 2], mode='markers', name=name,
                marker=dict(size=5, symbol=sym, opacity=0.85,
                            color='lightgrey' if lab == -1 else self.COLORS[int(lab) % len(self.COLORS)],
                            line=dict(width=0.5, color='black')),
                text=hover, hoverinfo='text'))

        centers = self._centroids_in_embedding(algorithm, method, 3)
        if centers is not None:
            fig.add_trace(go.Scatter3d(x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
                                       mode='markers', name='центры',
                                       marker=dict(size=12, color='black', symbol='diamond')))

        subtitle = ('цвет — кластер, форма — истинный класс' if self.y is not None
                    else 'цвет — кластер')
        fig.update_layout(
            title=f'{algorithm} в 3D-проекции {method} — {self.dataset_name}<br>'
                  f'<sup>{subtitle}</sup>',
            scene=dict(xaxis_title=f'{method} 1', yaxis_title=f'{method} 2',
                       zaxis_title=f'{method} 3'),
            width=950, height=750, legend=dict(itemsizing='constant'))
        if filename:
            fig.write_html(filename)
            print(f"Интерактивный график сохранён: {filename}")
        fig.show()
        return fig

    def plot_contingency(self, algorithm=None):
        """Таблица сопряжённости «истинный класс × кластер»."""
        if self.y is None:
            print('Истинные классы неизвестны — таблица сопряжённости недоступна.')
            return None
        algorithm = algorithm or self.best_algorithm()
        ct = pd.crosstab(pd.Series([self._class_label(c) for c in self.y], name='Класс'),
                         pd.Series(self.labels_[algorithm], name=f'Кластер ({algorithm})'))
        plt.figure(figsize=(1.2 * ct.shape[1] + 3, 0.8 * ct.shape[0] + 2))
        sns.heatmap(ct, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Сопоставление классов и кластеров ({algorithm})')
        plt.tight_layout()
        plt.show()
        return ct

    def plot_silhouette_diagram(self, algorithm='KMeans'):
        """Силуэтная диаграмма: значение s(i) для каждого объекта по кластерам."""
        labels = self.labels_[algorithm]
        mask = labels != -1
        X, labels = self.X[mask], labels[mask]
        s = silhouette_samples(X, labels)
        avg = s.mean()

        fig, ax = plt.subplots(figsize=(8, 6))
        y_lower = 10
        for lab in np.unique(labels):
            vals = np.sort(s[labels == lab])
            y_upper = y_lower + len(vals)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                             color=self.COLORS[int(lab) % len(self.COLORS)], alpha=0.8)
            ax.text(-0.05, y_lower + 0.5 * len(vals), str(lab))
            y_lower = y_upper + 10
        ax.axvline(avg, color='red', linestyle='--', label=f'средний силуэт = {avg:.3f}')
        ax.set_xlabel('Коэффициент силуэта s(i)')
        ax.set_ylabel('Объекты, сгруппированные по кластерам')
        ax.set_yticks([])
        ax.set_title(f'Силуэтная диаграмма — {algorithm}')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_cluster_profiles(self, algorithm=None, n_features=8):
        """Профили кластеров: средние z-оценки признаков в каждом кластере"""
        algorithm = algorithm or self.best_algorithm()
        labels = self.labels_[algorithm]
        Z = pd.DataFrame(StandardScaler().fit_transform(self.X_df), columns=self.feature_names)
        Z['cluster'] = labels
        profile = Z[Z['cluster'] != -1].groupby('cluster').mean()

        plt.figure(figsize=(max(10, 0.7 * len(self.feature_names)), 0.7 * len(profile) + 2))
        sns.heatmap(profile, annot=len(self.feature_names) <= 20, fmt='.2f',
                    cmap='RdBu_r', center=0)
        plt.title(f'Средние z-оценки признаков по кластерам ({algorithm})')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # Радар-чарт по признакам с наибольшим разбросом средних между кластерами
        top = profile.var(axis=0).sort_values(ascending=False).index[:n_features]
        angles = np.linspace(0, 2 * np.pi, len(top), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='polar'))
        for lab, row in profile[top].iterrows():
            color = self.COLORS[int(lab) % len(self.COLORS)]
            vals = row.tolist() + row.tolist()[:1]
            ax.plot(angles, vals, 'o-', linewidth=2, color=color, label=f'кластер {lab}')
            ax.fill(angles, vals, alpha=0.15, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(top, fontsize=9)
        ax.set_title(f'Профили кластеров ({algorithm})', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.show()
        return profile

    def plot_dendrogram(self, p=30):
        """Дендрограмма иерархической кластеризации (метод Уорда)."""
        Z = linkage(self.X, method='ward')
        plt.figure(figsize=(14, 5))
        dendrogram(Z, truncate_mode='lastp', p=p, leaf_rotation=90, show_contracted=True)
        if self.n_clusters:
            # высота, на которой дерево распадается ровно на n_clusters ветвей
            cut = (Z[-self.n_clusters, 2] + Z[-self.n_clusters + 1, 2]) / 2
            plt.axhline(cut, color='r', linestyle='--', label=f'разрез на {self.n_clusters} кластера')
            plt.legend()
        plt.title(f'Дендрограмма (Ward) — {self.dataset_name}')
        plt.xlabel('Объекты / размер свёрнутой ветви')
        plt.ylabel('Расстояние объединения')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Полный конвейер
    # ------------------------------------------------------------------
    def run_full_pipeline(self, n_clusters=None, embeddings=('PCA', 't-SNE', 'UMAP'),
                          interactive=True, html_file=None):
        """Полный цикл: анализ -> подбор k -> кластеризация -> оценка -> 2D/3D-визуализация."""
        print("=" * 70)
        print(f"КЛАСТЕРИЗАЦИЯ И ВИЗУАЛИЗАЦИЯ: {self.dataset_name}")
        print("=" * 70)
        self.describe_data()
        self.plot_eda()
        self.find_optimal_k()
        self.estimate_dbscan_eps()
        self.fit_clustering(n_clusters=n_clusters)
        if HAS_UMAP:
            self.fit_dbscan_on_embedding('UMAP', 2)
        display(self.evaluate_clustering().round(3))
        for method in embeddings:
            if method == 'UMAP' and not HAS_UMAP:
                continue
            self.plot_clusters_2d(method)
        self.plot_clusters_3d('PCA')
        if interactive:
            self.plot_interactive_3d_clusters('PCA', filename=html_file)
        self.plot_contingency()
        best = self.best_algorithm()
        self.plot_silhouette_diagram(best if not best.startswith('DBSCAN') else 'KMeans')
        self.plot_cluster_profiles()
        print(f"\nЛучший алгоритм: {best}")
        return self.metrics_
