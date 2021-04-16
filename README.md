# RandomForest-Decision-Tree

人工智慧概論課堂作業  
利用決策樹和隨機森林預測明日天氣。  
使用資料：kaggle Rain in Australia
https://www.kaggle.com/jsphyg/weather-dataset-rattle-package  

利用CART樹建構隨機森林。
演算法：
定義K個決策樹  
定義每一個決策樹之隨機樣本大小n

1.從資料集中取出大小為n的隨機樣本，取後放回
2.從選取之資料訓練決策樹，選出能讓訊息增益最多之特徵分割點做切割
3.重複k次步驟1,2
4.對於測試資料，利用多數決之方式，由k棵決策樹決定預測結果。

