# Getting Started

## Prerequisites
- Python 3.11+
- [Poetry](https://python-poetry.org/) for dependency management
- Git (for version control)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/trading-stock-agents.git
   cd trading-stock-agents
   ```

2. **Install dependencies**
   ```bash
   poetry install
   ```

3. **Configure your environment**
   - Copy `.env.example` to `.env`
   - Update the configuration as needed

## Basic Usage

### Fetching Data
To fetch data for a specific date:
```bash
poetry run python fetch_daily_data.py --date 20230707
```

### Running the Feature Pipeline
To process and enrich the data:
```python
from feature_engineer import FeatureEngineer

# Initialize the feature engineer
engineer = FeatureEngineer()

# Process a single file
engineer.enrich_file('data/AAPL/202307/data-20230707.csv')
```

### Viewing Results
- Processed data is saved in the same directory as the input files
- Charts are saved in the `charts` subdirectory
- Signal information is saved in `signals-YYYYMMDD.csv` files

## Configuration
Edit `config.py` to:
- Add/remove tickers
- Adjust date ranges
- Modify data storage locations

## Common Workflows

### Daily Update
1. Fetch new data
2. Process with feature engineering
3. Generate signals
4. Review charts and signals

### Historical Backfill
```bash
poetry run python backfill.py --start 20230101 --end 20230630
```

## Next Steps
- Explore the [Feature Engineering](features/overview.md) documentation
- Learn about [Model Training](models/overview.md)
- Check the [API Reference](api/overview.md) for advanced usage
