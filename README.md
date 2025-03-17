# 🍔 Amazon Food Reviews Sentiment Analysis

## 📌 專案簡介
本專案針對 Amazon 美食評論進行情緒分類，將評論分為正向（Positive）或負向（Negative）。  
專案採用 Python 與機器學習方法，利用兩種文字向量化技術（TF-IDF、Word2Vec）搭配隨機森林（Random Forest）進行模型訓練與預測，並透過 K-Fold Cross-Validation 評估模型效能。

## 🔧 使用技術
  - Pandas (Data processing and manipulation)
  - Scikit-learn（Machine learning algorithms, feature vectorization, cross-validation）
  - Gensim（Word2Vec向量模型）
  - 正規表示式（re）、文字處理（string）

## 💻Project Source Codes:
[Amazon-Fine-Food-Reviews](https://github.com/thegloriachen/Amazon-Fine-Food-Reviews/blob/main/Amazon-Fine-Food-Reviews.py)

## 🚀 專案流程說明
### 1️⃣ 資料預處理（Data Preprocessing）
- 讀取 `Reviews.csv` 檔案，取前 10000 筆樣本  
- 篩選欄位：`Text`（評論內容）、`Score`（評論評分）  
- 將 `Score` 轉為二元標籤：  
  - `Score >= 4` ➜ 正向評論（1）  
  - `Score < 4` ➜ 負向評論（0）  
- 文本清理（文字正規化）：  
  - 轉小寫  
  - 移除 HTML 標籤、特殊符號、數字  
  - 去除英文停用詞與無意義單字  
- 最終將處理後的文本作為訓練資料

### 2️⃣ 特徵工程（Feature Engineering）
#### 📝 方法一：TF-IDF 向量化
- 使用 `CountVectorizer` 建立詞頻矩陣  
- 搭配 `TfidfTransformer` 計算 TF-IDF 權重向量  
- 輸出 TF-IDF 特徵矩陣供模型訓練使用

#### 📝 方法二：Word2Vec 向量化
- 利用 `gensim.models.Word2Vec` 訓練詞向量模型  
- 每篇評論取單詞向量平均值作為評論向量  
- 輸出 Word2Vec 向量資料供模型訓練

### 3️⃣ 模型建置（Model Building）
- 採用 `RandomForestClassifier` 隨機森林模型  
- 訓練與測試資料集比例 ➜ 75% 訓練 / 25% 測試  
- 設定參數：  
  - `n_estimators=100`  
  - `random_state=42`（確保結果可重現）

### 4️⃣ 模型評估（Evaluation）
- 指標：  
  - Accuracy（準確率）  
- 方法：  
  - 標準測試集評估  
  - K-Fold Cross Validation ➜ `cv=4`

| 模型類型       | 測試集準確率 | 交叉驗證平均準確率 |
|---------------|-------------|------------------|
| TF-IDF        | 82.56%      | 80.11%  
| Word2Vec      | 78.32%      | 76.99%

---

## 📈 結果展示
- TF-IDF 模型  
  - 測試集準確率：`82.56%`  
  - K-Fold CV 平均準確率：`80.11%`  
- Word2Vec 模型  
  - 測試集準確率：`78.32%`  
  - K-Fold CV 平均準確率：`76.99%` 
