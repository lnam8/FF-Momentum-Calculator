The Python file in this directory will call the Sleeper API to pull the necessary data required for the 6 players required and append the data to `weekly_data.csv`. The script will also pull the Wide Receivers(WR), Running Backs(RB), and Tight Ends(TE) who scored during that week and also append it to the same CSV file.

This file should be run once a week when all of the games for the preceeding week are complete

The players that these projections will be populated for are:
- Puka Nacua
- Kyle Pitts
- Josh Jacobs
- Calvin Ridley
- Raheem Mostert
- Zach Ertz

## Usage

```bash
python3 weekly_data.py --week <week>
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)