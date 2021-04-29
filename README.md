# RandomForest-DecisionTree
Supervised Learning algorithm practice
利用決策樹和隨機森林預測明日天氣，並利用f1-score來評定最後結果。  
*F1-score
針對資料級label若分配不均(例如label為1之data佔總data 90%，label為0之data佔總data 10%)
若利用Accuracy來判定準確率，無法有效衡量模型好壞。

使用資料：kaggle Rain in Australia
https://www.kaggle.com/jsphyg/weather-dataset-rattle-package  


基於CART決策樹實線之隨機森林。  
演算法：  
定義K個決策樹  
定義每一個決策樹之隨機樣本大小n  
  
1.從資料集中取出大小為n的隨機樣本，取後放回  
2.從選取之資料訓練決策樹，選出能讓訊息增益最多之特徵分割點做切割  
3.重複k次步驟1,2  
4.對於測試資料，利用多數決之方式，由k棵決策樹決定預測結果。  

為避免過度擬合而降低預測準確性，加入預剪枝方法。
若在決策過程中發現分裂節點後之訊息增益降低，則停止剪枝。
  
======result======  
實作之在隨機抽樣訓練模型，並利用Testing Data測試達到f1-score=0.6以上之結果




