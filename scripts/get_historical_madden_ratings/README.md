This script pulls historical weekly Madden ratings from years 2022 - 2023. API does not work for years before 2022. This script pulls historical data and only needs to be run once. Directory for output is "./historical_weekly_ratings"

### Install Dependencies 
1. pip install -r requirements.txt

### Run Script 
``` python main.py --dir [dir] ``` 

Example 
```python main.py --dir ./weekly_madden_ratings```

```python main.py --dir /Users/lnam/Desktop/Madden```

Files will be created to specified directory as year_{year}_week_{week}_madden_ratings.csv