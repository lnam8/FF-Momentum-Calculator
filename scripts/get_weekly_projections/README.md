The Python files in this package will append `projection_data.csv` with weekly projection point values from either ESPN, Sleeper, or FantasyPros.

The players that these projections will be populated for are:
- Puka Nacua
- Kyle Pitts
- Josh Jacobs
- Calvin Ridley
- Raheem Mostert
- Zach Ertz

## Install Dependencies

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required package.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 espn.py --week <week>

python3 fantasypros.py --week <week>

python3 sleeper.py --week <week>
```

The resulting data, if any, will be appended to the CSV file `projection_data.csv`.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)