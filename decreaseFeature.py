import os
import subprocess
from collections import Counter
import nltk

# =====================================================================
# 解決問題 3：NLTK 新舊版本模型名稱相容性修正
# =====================================================================
try:
    # 嘗試下載新版模型名稱
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    # 如果環境仍是舊版 NLTK，則下載舊版名稱作為後備
    nltk.download("averaged_perceptron_tagger", quiet=True)
except Exception:
    pass

nltk.download("punkt", quiet=True)
nltk.download("universal_tagset", quiet=True)


def safe_pos_tag(words):
    """
    安全呼叫詞性標註的封裝函式，自動切換新舊版模型名稱
    """
    try:
        # 先嘗試新版規範
        return nltk.pos_tag(words, tagset="universal", lang="eng")
    except Exception:
        try:
            # 失敗則退回傳統規範
            return nltk.pos_tag(words, tagset="universal")
        except Exception as e:
            print(f"【NLTK 警告】詞性標註失敗，將全數視為名詞處理。錯誤訊息: {e}")
            return [(w, "NOUN") for w in words]


# =====================================================================
# 核心功能：結合 C++ 計算並自動命名
# =====================================================================
def generate_word_to_label_mapping(unique_descriptions):
    """
    將不重複的品名清單傳給 C++ 進行圖形分群，並利用 Python NLTK 命名
    """
    print("【Python】正在生成 C++ 所需的暫存檔...")
    # 將不重複品名寫入暫存檔，供 C++ 讀取
    with open("temp_unique_desc.txt", "w", encoding="utf-8") as f:
        for desc in unique_descriptions:
            f.write(f"{desc}\n")

    # 自動編譯與執行 C++ 程式碼
    cpp_source = "clusterWords.cpp"
    cpp_executable = "./clusterWords" if os.name != "nt" else "clusterWords.exe"

    print("【Python】正在編譯 C++ 核心分群程式...")
    compile_cmd = ["g++", "-O3", cpp_source, "-o", cpp_executable]
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"C++ 編譯失敗！請確認系統已安裝 g++。\n錯誤訊息: {result.stderr}")

    print("【Python】正在執行 C++ 高速社群發現...")
    run_result = subprocess.run([cpp_executable], capture_output=True, text=True)
    if run_result.returncode != 0:
        raise RuntimeError(f"C++ 執行期間出錯！\n錯誤訊息: {run_result.stderr}")

    # 讀取 C++ 算好的分群結果
    print("【Python】正在讀回 C++ 分群結果並進行特徵篩選...")
    if not os.path.exists("temp_partition.txt"):
        raise FileNotFoundError("找不到 C++ 輸出的暫存結果 temp_partition.txt")

    community_to_words = {}
    global_word_counts = {}

    with open("temp_partition.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                word, community_id, count = parts[0], int(parts[1]), int(parts[2])
                global_word_counts[word] = count
                if community_id not in community_to_words:
                    community_to_words[community_id] = []
                community_to_words[community_id].append(word)

    # 利用 NLTK 詞性過濾，為每一群賦予最合適的核心名詞 Label
    community_to_label = {}
    for community_id, words in community_to_words.items():
        tagged_words = safe_pos_tag(words)

        # 核心規則：只留名詞 (NOUN)，自動點名淘汰形容詞與顏色
        noun_words = [word for word, tag in tagged_words if tag == "NOUN"]
        if not noun_words:
            noun_words = words  # 容錯

        # 依據出現總頻率由大到小排序
        top_nouns = sorted(noun_words, key=lambda w: global_word_counts.get(w, 0), reverse=True)
        best_words = top_nouns[:2]

        # 調整語序，使其貼近原始電商商品習慣
        sample_names = [name for name in unique_descriptions if all(w in name for w in best_words)]
        if sample_names and len(best_words) == 2:
            first_match = sample_names[0].split()
            best_words = sorted(best_words, key=lambda w: first_match.index(w) if w in first_match else 0)

        community_to_label[community_id] = " ".join(best_words)

    # 建立「單字 -> 新標籤」的字典
    word_to_label = {}
    with open("temp_partition.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                word, community_id = parts[0], int(parts[1])
                word_to_label[word] = community_to_label[community_id]

    # 清理中途產生的暫存檔
    for temp_file in ["temp_unique_desc.txt", "temp_partition.txt", cpp_executable]:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass

    return word_to_label