import os
from collections import Counter
import pandas as pd
# 引入剛才建立的過濾函式
from decreaseFeature import generate_word_to_label_mapping


# =====================================================================
# 模擬簡化版的 HP 樹結構建立與輸出 (對應第4點需求)
# =====================================================================
class HpNode:
    def __init__(self, name, count=0):
        self.name = name
        self.count = count
        self.children = {}


def build_and_save_hp_tree(df_clean, tree_output_file):
    """
    利用 InvoiceDate (日期/發票) 和 new_label (商品類別) 來建立並輸出樹結構
    """
    print("【主程式】正在依據交易群組建立 HP 樹結構...")

    # 建立樹根
    root = HpNode("Root")

    # 按 InvoiceDate 分組，把同一天的所有 new_label 視為一次頻繁項目的集合
    grouped = df_clean.groupby("InvoiceDate")["new_label"].apply(list)

    for transaction in grouped:
        # 計算交易內品名頻率並排序 (HP樹標準做法：高頻的靠在樹根上層)
        counts = Counter(transaction)
        sorted_items = [item for item, _ in counts.most_common()]

        # 逐一將項目掛載到樹枝上
        current_node = root
        for item in sorted_items:
            if item not in current_node.children:
                current_node.children[item] = HpNode(item, 1)
            else:
                current_node.children[item].count += 1
            current_node = current_node.children[item]

    # 將樹結構以階層縮排格式寫入文字檔
    print(f"【主程式】正在輸出樹結構檔案至 {tree_output_file}...")
    with open(tree_output_file, "w", encoding="utf-8") as f:
        def _write_node(node, depth):
            indent = "  " * depth
            if node.name != "Root":
                f.write(f"{indent}|-- {node.name} (support: {node.count})\n")
            else:
                f.write("HP-Tree Root\n")
            for child in node.children.values():
                _write_node(child, depth + 1)

        _write_node(root, 0)
    print("【主程式】樹結構檔案輸出完成！")


# =====================================================================
# 主排程流程
# =====================================================================
def main():
    input_filename = "online_retail_II.csv"
    csv_output_filename = "new_label.csv"
    tree_output_filename = "hp_tree_structure.txt"

    if not os.path.exists(input_filename):
        print(f"錯誤：在當前目錄找不到 {input_filename}，請先確認檔案位置。")
        return

    print("【主程式】1. 正在讀取原始電商檔案...")
    # 載入研究所需的三個核心特徵欄位
    try:
        df = pd.read_csv(input_file, usecols=["Description", "InvoiceDate"])
    except Exception:
        df = pd.read_csv(input_filename)
        df.columns = df.columns.str.strip()
        df = df[["Description", "InvoiceDate"]]

    # 2. 資料清洗
    df = df.dropna(subset=["Description", "InvoiceDate"])
    df["Description"] = df["Description"].astype(str).str.strip().str.upper()
    df["InvoiceDate"] = df["InvoiceDate"].astype(str).str.strip()

    # 3. 提取不重複商品名稱以供 C++ 加速計算
    unique_descriptions = df["Description"].drop_duplicates().tolist()
    print(f"【主程式】不重複商品名稱共計 {len(unique_descriptions)} 筆。")

    # 4. 呼叫 decreaseFeature 中的跨語言分群模組
    word_to_label = generate_word_to_label_mapping(unique_descriptions)

    # 5. 將分群標籤映射回 DataFrame
    print("【主程式】2. 正在對原始資料進行特徵壓縮與標籤映射...")

    def assign_label(desc):
        words = desc.split()
        labels = [word_to_label[w] for w in words if w in word_to_label]
        if labels:
            return Counter(labels).most_common(1)[0][0]
        return "OTHER"

    # 為求精確且快速，先對不重複項進行 Mapping 再 Merge 回主表
    mapping_df = pd.DataFrame({"Description": unique_descriptions})
    mapping_df["new_label"] = mapping_df["Description"].apply(assign_label)

    # 合併回原表，此時 df 將同時擁有 description, invoiceDate, new_label 三個特徵
    df_final = pd.merge(df, mapping_df, on="Description", how="left")

    # 調整欄位名稱順序符合需求：description, invoiceDate, new_label
    df_final = df_final[["Description", "InvoiceDate", "new_label"]]

    # 6. 輸出新 CSV 檔案 (滿足需求 1)
    print(f"【主程式】正在輸出特徵壓縮後的檔案至 {csv_output_filename}...")
    df_final.to_csv(csv_output_filename, index=False, encoding="utf-8")

    # 7. 建立 HP 樹結構並導出文字檔 (滿足需求 4)
    build_and_save_hp_tree(df_final, tree_output_filename)
    print(" 全自動化跨語言關聯分析流程執行成功！")


if __name__ == "__main__":
    main()