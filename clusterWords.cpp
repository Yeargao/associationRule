#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <string>
#include <unordered_map>
using namespace std;

int nowID = 0;

struct hpNode {
    int code = 0;
    int times = 0;
    int ID = 0;
    hpNode* parent = nullptr;
    hpNode* cousin = nullptr;
    vector<hpNode*> allChild; // 只儲存真正擁有的子節點
};

struct invoice {
    int productKinds;
    unordered_map<string, bool> buyWhat;
    vector<int> allCode; // 儲存過濾並排序後的商品編碼
};

bool Compare(const pair<string, int>& p1, const pair<string, int>& p2) {
    return p1.second > p2.second;
}

// 【新增 Function】: 尋找或建立子節點，解決重複程式碼
hpNode* findOrAddChild(hpNode* parentNode, int targetCode, int& idCounter) {
    hpNode* childNode = nullptr;

    // 檢查目前節點的小孩中，是否已經有這個 code
    for (auto child : parentNode->allChild) {
        if (child->code == targetCode) {
            childNode = child;
            break;
        }
    }

    if (childNode != nullptr) {
        // 小孩已存在，次數 +1
        childNode->times++;
    } else {
        // 小孩不存在，建立新節點
        childNode = new hpNode{ targetCode, 1, idCounter++, parentNode, nullptr };
        parentNode->allChild.push_back(childNode);
    }
    return childNode;
}

void outTree(hpNode* hpn);

int main() {
    string date, Havesecond, Description;
    int produceKind = 0, nowAt = 0;
    unordered_map<string, int> productNum;
    unordered_map<string, invoice*> eachInvoice;
    unordered_map<string, int> productCode;

    // 測試資料讀取 (限制 22 次)
    int ii = 0;
    while (ii++ <= 22) {
        if (!(cin >> Description >> date >> Havesecond)) break;
        if (productNum[Description] == 0) produceKind++;
        productNum[Description]++;
        string newDandT = date + " " + Havesecond;

        if (!eachInvoice[newDandT]) eachInvoice[newDandT] = new invoice{ 0 };
        eachInvoice[newDandT]->productKinds++;
        eachInvoice[newDandT]->buyWhat[Description] = true;
    }

    // 篩選次數 > 1 的商品
    vector<pair<string, int>> toCompareProduct;
    for (const auto& p : productNum) {
        if (p.second > 1) toCompareProduct.push_back({ p.first, p.second });
    }
    sort(toCompareProduct.begin(), toCompareProduct.end(), Compare);

    // 商品編碼 (從 1 開始)
    for (const auto& p : toCompareProduct) productCode[p.first] = ++nowAt;

    // 用來記錄各個 code 的最後一個節點，方便串接 cousin
    vector<hpNode*> allCodeNode(nowAt + 1, nullptr);
    vector<hpNode*> allFirstCodeNode(nowAt + 1, nullptr); // 大小修正為 nowAt + 1
    vector<hpNode*> allCodeRoot(nowAt + 1, nullptr);      // 大小修正為 nowAt + 1
    hpNode* hpRoot = new hpNode{ 0, 0, 0, nullptr, nullptr };

    // 開始建樹
    for (const auto& eiPair : eachInvoice) {
        invoice* ei = eiPair.second;

        // 1. 篩選出這張發票中「有進入排名(次數>1)」的商品
        for (const auto& p : ei->buyWhat) {
            if (productCode.count(p.first)) {
                ei->allCode.push_back(productCode[p.first]);
            }
        }
        // 2. 排序 (依照編碼由小到大，即次數由大到小)
        sort(ei->allCode.begin(), ei->allCode.end());

        hpNode* nowNode = hpRoot;

        // 3. 逐一將商品加入樹中
        for (int codes : ei->allCode) {
            bool isNewNode = (allCodeNode[codes] == nullptr);

            // 呼叫重構後的函式
            nowNode = findOrAddChild(nowNode, codes, nowID);

            // 如果是新建立的節點，串接 cousin 鏈
            if (isNewNode || allCodeNode[codes]->cousin == nowNode) {
                if (allCodeNode[codes] == nullptr) {
                    allFirstCodeNode[codes] = nowNode;
                } else if (allCodeNode[codes] != nowNode) {
                    allCodeNode[codes]->cousin = nowNode;
                }
                allCodeNode[codes] = nowNode;
            }
        }
    }

    // 建立每種商品的獨立樹
    for (const auto& p : allFirstCodeNode) {
        if (!p) continue; // 略過沒用到的 code

        int currentCode = p->code;
        allCodeRoot[currentCode] = new hpNode{ currentCode, 0, nowID++, nullptr, nullptr };

        hpNode* nowCousin = p;
        while (nowCousin) {
            hpNode* nowNode = nowCousin->parent; // 從 parent 開始往上回溯
            hpNode* subTreeParent = allCodeRoot[currentCode]; // 子樹的跟隨指標

            // 收集祖先路徑 (因為要由上往下建，先用 vector 存起來)
            vector<int> ancestorCodes;
            while (nowNode && nowNode->code != 0) {
                ancestorCodes.push_back(nowNode->code);
                nowNode = nowNode->parent;
            }

            // 由上往下建立該條件路徑的子樹
            reverse(ancestorCodes.begin(), ancestorCodes.end());
            for (int ancCode : ancestorCodes) {
                subTreeParent = findOrAddChild(subTreeParent, ancCode, nowID);
            }

            nowCousin = nowCousin->cousin;
        }

        // 輸出該商品的獨立樹
        cout << "\n=== Tree for Product Code: " << currentCode << " ===\n";
        cout << left << setw(8) << "ID" << setw(8) << "code" << setw(8) << "times" << setw(12) << "parent_ID" << '\n';
        outTree(allCodeRoot[currentCode]);
    }

    return 0;
}

// Pre-order 輸出樹狀結構
void outTree(hpNode* hpn) {
    if (!hpn) return;

    // 印出當前節點資訊
    cout << left << setw(8) << hpn->ID
         << setw(8) << hpn->code
         << setw(8) << hpn->times;
    if (hpn->parent) {
        cout << left << setw(12) << hpn->parent->ID;
    } else {
        cout << left << setw(12) << "NULL";
    }
    cout << '\n';

    // 遞迴走訪所有子節點
    for (const auto& child : hpn->allChild)
        outTree(child);
}
