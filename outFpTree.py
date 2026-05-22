import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             adjusted_rand_score, silhouette_score)
import hdbscan
import umap
import matplotlib.font_manager as fm


# ==========================================
# 程式流程備註 (AI 產生指令參考):
# 1. 環境設定: 支援中文字體顯示與繪圖風格設定。
# 2. 數據處理: 載入 CSV 或模擬數據，進行類別標籤數值化 (LabelEncoding) 與特徵標準化 (Standard Scaling)。
# 3. 降維與分群: 使用 UMAP 將高維數據降至 2 維，隨後以 HDBSCAN 進行自動分群。
# 4. 標籤映射: 將分群結果 (Cluster) 與原始人格標籤 (Ground Truth) 進行眾數比對，標記為內向或外向。
# 5. 模型訓練: 使用隨機森林 (Random Forest) 進行分類訓練。
# 6. 指標評估:
#    - 分類指標: 計算測試集的 Accuracy, Precision, Recall, F1-score。
#    - 分群指標: 計算與真實標籤對比的 Accuracy, ARI (調整蘭德係數) 以及 Silhouette Score (輪廓係數)。
# 7. 多維視覺化: 包含 UMAP 分布圖、特徵雷達圖 (存檔)、相關性熱圖 (存檔)、特徵重要性圖、HDBSCAN 樹狀圖與最小生成樹 (MST)。
# ==========================================

# ==========================================
# 0. 中文字體支援處理
# ==========================================
def get_chinese_font():
    # 嘗試尋找系統中的中文字體，確保圖表能顯示中文
    fonts = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'Heiti TC', 'Sans-serif']
    for f in fonts:
        if f in [font.name for font in fm.fontManager.ttflist]:
            return f
    return 'DejaVu Sans'


plt.rcParams['font.sans-serif'] = [get_chinese_font()]
plt.rcParams['axes.unicode_minus'] = False  # 解決座標軸負號顯示問題
sns.set_theme(style="whitegrid", font=get_chinese_font())


# ==========================================
# 1. 數據載入與預處理
# ==========================================
def load_and_preprocess(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        # 若無檔案則建立模擬數據
        print("找不到檔案，建立模擬數據...")
        np.random.seed(42)
        n = 400
        data = np.random.rand(n, 5)
        df = pd.DataFrame(data, columns=['Social_Activity', 'Anxiety', 'Energy_Level', 'Stage_Fear', 'Social_Drain'])
        df['Personality'] = np.random.choice(['Introvert', 'Extrovert'], n)
        # 根據人格調整模擬數據特徵
        df.loc[df['Personality'] == 'Introvert', 'Social_Activity'] -= 0.3
        df.loc[df['Personality'] == 'Extrovert', 'Social_Activity'] += 0.3

    # 將文字標籤 (Yes/No) 轉換為 0 或 1
    binary_map = {'Yes': 1, 'No': 0, 'True': 1, 'False': 0, True: 1, False: 0}
    cols_to_fix = ['Stage_Fear', 'Social_Drain', 'Stage_fear', 'Drained_after_socializing']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(0).astype(int)

    # 將目標人格轉換為數值：內向=0，外向=1
    personality_map = {'Introvert': 0, 'Extrovert': 1}
    if 'Personality' in df.columns:
        df['Personality_Encoded'] = df['Personality'].map(personality_map).fillna(0).astype(int)

    le = LabelEncoder()
    le.fit(['Introvert', 'Extrovert'])  # 固定編碼順序
    return df, le


df, le = load_and_preprocess('personality_dataset.csv')
# 選取數值特徵作為 X，人格編碼作為 y (真值)
X_numeric = df.select_dtypes(include=[np.number]).drop(['Personality_Encoded'], axis=1, errors='ignore')
y_true = df['Personality_Encoded']

# ==========================================
# 2. UMAP 降維與 HDBSCAN 分群
# ==========================================
# 數據標準化：讓不同單位的特徵具有相同的權重
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# UMAP 降維：將多維特徵壓縮到 2 維以利分群
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(X_scaled)

# HDBSCAN 分群：基於密度的自動分群演算法
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5, gen_min_span_tree=True)
df['HDBSCAN_Cluster'] = clusterer.fit_predict(embedding)

# ==========================================
# 3. 標籤映射與分析
# ==========================================
cluster_mapping = {}
unique_clusters = sorted(df['HDBSCAN_Cluster'].unique())
df['Cluster_Type_Encoded'] = -1

for cluster in unique_clusters:
    if cluster == -1:  # HDBSCAN 的 -1 代表雜訊 (Noise)
        cluster_mapping[cluster] = "Noise"
        continue

    # 找出該群體中最主流的人格作為該群的標籤
    cluster_mask = df['HDBSCAN_Cluster'] == cluster
    mode_val_encoded = df[cluster_mask]['Personality_Encoded'].mode()[0]
    mode_val_name = "外向" if mode_val_encoded == 1 else "內向"

    cluster_mapping[cluster] = f"C{cluster} ({mode_val_name})"
    df.loc[cluster_mask, 'Cluster_Type_Encoded'] = mode_val_encoded

df['Cluster_Label'] = df['HDBSCAN_Cluster'].map(cluster_mapping)

# ==========================================
# 4. 指標計算
# ==========================================
# 分類模型：使用隨機森林預測人格
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y_true, test_size=0.2, random_state=42)
rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
y_rf_pred = rf.predict(X_test)

# 分群評估：排除雜訊點後進行比對
valid_mask = df['HDBSCAN_Cluster'] != -1
y_true_valid = df.loc[valid_mask, 'Personality_Encoded']
y_cluster_pred = df.loc[valid_mask, 'Cluster_Type_Encoded']

print("\n" + "=" * 30)
# 1. 分類指標：評估模型在未知數據上的預測能力
print("1. 分類指標 (Random Forest Test Set)")
print(f"Accuracy (準確率)  : {accuracy_score(y_test, y_rf_pred):.3f}")
print(f"Recall (召回率)    : {recall_score(y_test, y_rf_pred):.3f}")
print(f"Precision (精確率) : {precision_score(y_test, y_rf_pred):.3f}")
print(f"F1-score (F1分數)  : {f1_score(y_test, y_rf_pred):.3f}")

# 2. 分群指標：評估分群結果與真實標籤的一致性
print("\n2. 分群指標 (HDBSCAN vs Ground Truth)")
print(f"Cluster Accuracy (分群準確度): {accuracy_score(y_true_valid, y_cluster_pred):.3f}")
print(f"Adjusted Rand Index (ARI): {adjusted_rand_score(y_true, df['HDBSCAN_Cluster']):.3f}")
# Silhouette Score：衡量群內緊湊度與群間分離度
s_score = silhouette_score(X_scaled, df['HDBSCAN_Cluster'])
print(f"Silhouette Score (輪廓係數): {s_score:.3f}")
print("=" * 30)

# ==========================================
# 5. 視覺化：UMAP 對照圖
# ==========================================
plt.figure(figsize=(9, 7))
sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=df['Personality'], palette='Set1', s=50, alpha=0.6)
plt.title('Original Labels (Intro/Extro)')
plt.savefig("original_umap.png")
plt.show()

plt.figure(figsize=(9, 7))
sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=df['Cluster_Label'], palette='tab20', s=50)
plt.title('HDBSCAN Clusters (Sub-groups)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("hdbscan_umap.png")
plt.show()

# ==========================================
# 6. 雷達圖
# ==========================================
# 計算每一群在各特徵上的平均值，用於繪製雷達圖比較特徵差異
cluster_summary = df[valid_mask].groupby('Cluster_Label')[X_numeric.columns].mean()


def plot_radar_chart(df_summary):
    labels = df_summary.columns.tolist()
    num_vars = len(labels)
    # 計算雷達圖的角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 閉合多邊形

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for index, row in df_summary.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1]  # 閉合多邊形
        ax.plot(angles, values, linewidth=2, label=index)
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    plt.title('各群特徵平均值雷達圖對比', y=1.08, fontsize=15)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # 新增：儲存雷達圖
    plt.savefig("雷達圖.png")
    plt.show()


if not cluster_summary.empty:
    plot_radar_chart(cluster_summary)


# ==========================================
# 7. 特徵分析
# ==========================================

def plot_correlation_heatmap(df_full):
    # 繪製特徵相關性熱圖
    plot_df = df_full.select_dtypes(include=[np.number]).copy()
    if 'Personality_Encoded' in plot_df.columns:
        plot_df = plot_df.rename(columns={'Personality_Encoded': '人格(內0/外1)'})

    plt.figure(figsize=(10, 8))
    corr_matrix = plot_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                fmt=".2f", linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('相關性熱圖', fontsize=15, pad=20)
    plt.savefig("相關性熱圖.png")
    plt.show()


def plot_feature_importance_bar(model, feature_names):
    # 顯示隨機森林的特徵重要性
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance Score')
    plt.title('特徵重要性熱圖', fontsize=14)
    for i, v in enumerate(importances[indices]):
        plt.text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig("特徵重要性熱圖.png")
    plt.show()


plot_correlation_heatmap(df)
plot_feature_importance_bar(rf, X_numeric.columns)


# ==========================================
# 8. HDBSCAN 樹狀圖與 MST
# ==========================================
def plot_hdbscan_tree_custom(clusterer, df_source):
    # 繪製凝聚樹狀圖
    plt.figure(figsize=(12, 8))
    tree = clusterer.condensed_tree_
    tree.plot(select_clusters=True, selection_palette=sns.color_palette('deep', 8))

    chosen_clusters = clusterer.condensed_tree_._select_clusters()
    cluster_info = ""

    for i, c_id in enumerate(chosen_clusters):
        c_mask = (df_source['HDBSCAN_Cluster'] == i)
        count = c_mask.sum()
        if count > 0:
            mode_val = df_source[c_mask]['Personality_Encoded'].mode()[0]
            label = "外向" if mode_val == 1 else "內向"
            cluster_info += f"群組{i}: {count}筆 ({label})\n"

    plt.annotate(cluster_info, xy=(0.02, 0.7), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                 fontsize=10)

    plt.title(f'樹狀圖 (分出 {len(chosen_clusters)} 群)', fontsize=14)
    plt.savefig("樹狀圖.png")
    plt.show()

    # 繪製最小生成樹 (MST)
    plt.figure(figsize=(12, 6))
    clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6,
                                          node_size=10, edge_linewidth=0.5)
    plt.title('最小生成樹', fontsize=14)
    plt.savefig("最小生成樹.png")
    plt.show()


plot_hdbscan_tree_custom(clusterer, df)