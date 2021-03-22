# Electricity Forecasting
Members: 方郁文、陳香君

## Goal 
Predict the operating reserve (備轉容量) value from 2021/03/23 to 2021/03/29.

## Dataset
在 Dataset/ 底下有 electricity 和 weather ，以下分別介紹：
1. electricity/ 底下包含 「2020年度每日尖峰備轉容量率.csv」和 「2021年度每日尖峰備轉容量率.csv」<br>
皆是從作業說明中的 "台灣電⼒公司_本年度每⽇尖峰備轉容量率"  下載來的<br>
https://data.gov.tw/dataset/19995
3. weather/ 底下包含 "weather_day.csv" 和 "weather_forecast.csv" <br>
其中"weather_forecast.csv" 來自作業說明中的"未來一週天氣預報"，並加以整理成csv檔案 <br>
而"weather_day.csv"來自"一年觀測資料-局屬地面測站觀測資料"(https://opendata.cwb.gov.tw/dataset/climate/C-B0024-002) <br>
並自行整理成".csv"

## Method
使用NN
