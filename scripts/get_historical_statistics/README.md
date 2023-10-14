The Python file in this directory will pull the weekly statistics of Wide Receivers(WR), Running Backs(RB), and Tight Ends(TE) who scored during each week from the years 2002 through 2022 and also write each data to a new CSV file.

This file is ran only once as this data is purely historical.  

One caveat of this script is that the Team Name for each player retrieved is the team that the player is on in the year 2023 if that player is on a team at all.  
Example:  
Year 2019 - Week 1: Deandre Hopkins shows team Tennessee  
In 2019, Deandre Hopkins played for the Houston Texans

## Usage

```bash
python3 historical_weekly_data.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)