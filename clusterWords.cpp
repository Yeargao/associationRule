#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>

using namespace std;

int main() {
    // 讀取由 Python 生成的不重複商品名稱暫存檔
    ifstream infile("temp_unique_desc.txt");
    if (!infile.is_open()) {
        cerr << "【C++ 錯誤】無法開啟暫存檔 temp_unique_desc.txt！" << endl;
        return 1;
    }

    map<string, int> word_counts;
    map<pair<string, string>, int> co_occurrence;
    set<string> unique_words;
    string line;

    // 1. 讀取品名並計算共現次數
    while (getline(infile, line)) {
        stringstream ss(line);
        string word;
        vector<string> words;
        while (ss >> word) {
            words.push_back(word);
            unique_words.insert(word);
            word_counts[word]++;
        }

        for (size_t i = 0; i < words.size(); ++i) {
            for (size_t j = i + 1; j < words.size(); ++j) {
                string w1 = words[i];
                string w2 = words[j];
                if (w1 > w2) swap(w1, w2);
                co_occurrence[{w1, w2}]++;
            }
        }
    }
    infile.close();

    // 2. 基礎社群聚合演算法 (Fast Greedy Component Search)
    map<string, int> word_to_community;
    int community_id = 0;
    for (const auto& word : unique_words) {
        word_to_community[word] = community_id++;
    }

    // 依據共現權重由高到低排序，優先合併強關聯字詞
    vector<pair<pair<string, string>, int>> edges(co_occurrence.begin(), co_occurrence.end());
    sort(edges.begin(), edges.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    // 開始聚合連線
    for (const auto& edge : edges) {
        string w1 = edge.first.first;
        string w2 = edge.first.second;
        int weight = edge.second;

        // 設定門檻值：共同出現超過 1 次才考慮合併，避免雜訊
        if (weight <= 1) break;

        int c1 = word_to_community[w1];
        int c2 = word_to_community[w2];

        if (c1 != c2) {
            // 將所有屬於 c2 幫派的字全部併入 c1
            for (auto& item : word_to_community) {
                if (item.second == c2) {
                    item.second = c1;
                }
            }
        }
    }

    // 3. 將分群結果輸出給 Python 讀取 (格式: 字詞 群組ID 總出現次數)
    ofstream outfile("temp_partition.txt");
    for (const auto& item : word_to_community) {
        outfile << item.first << " " << item.second << " " << word_counts[item.first] << "\n";
    }
    outfile.close();

    return 0;
}
