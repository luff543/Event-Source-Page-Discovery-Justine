# Generate dataset
這個部分是資料集生成，主要會從google搜尋引擎下關鍵字並將每一頁上的網站都收集下來

## Folder Structure
```
|--- generate_dataset
|	|--- engines
|	|--- pick_rank
|	|--- eventsource_展覽.csv
|	|--- eventsource_最新活動.csv
|	|--- eventsource_活動.csv
|	|--- eventsource_活動講座.csv
|	|--- eventsource_講座.csv
```

## 爬蟲
```
python google_search.py
run group.ipynb ->把不同關鍵字收尋到的結果整合
```
## download html
```
python from_homepage_dataset_save_page_new.py
```
會產生兩種html(original,with hidden elements)、一個資料夾裡面有不同類型的html檔和網頁截圖
最後是使用with hidden elements，file中的html檔是備份
