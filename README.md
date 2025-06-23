# NBA Finals 2025 Analysis Tool

This comprehensive analysis tool analyzes NBA Finals 2025 statistics to determine MVP, All-Finals Team, and All-Defensive Team selections based on deep statistical analysis.

## Files Description

### Main Analysis Tools
- **`FINALS_2025_TOOL.py`** - Complete interactive analysis tool with visualizations
- **`quick_analysis.py`** - Simplified version that runs analysis automatically
- **`export_csv.py`** - Simple script to export all data to CSV files

### Data Files
- **`FINALS_STATS_2025 - sportsref_download.xls.csv`** - Raw Finals statistics data
- **`requirements.txt`** - Required Python packages

## How to Use

### Option 1: Full Interactive Tool
```bash
python FINALS_2025_TOOL.py
```
This runs the complete tool with:
- Comprehensive analysis report
- Interactive menu options
- Visualization generation (if matplotlib available)
- CSV export functionality

### Option 2: Quick Analysis
```bash
python quick_analysis.py
```
This automatically generates:
- Complete analysis report
- Text file output
- Basic CSV exports

### Option 3: CSV Export Only
```bash
python export_csv.py
```
This only exports data to CSV files without running full analysis.

## Analysis Features

### Finals MVP Selection
Based on weighted scoring system considering:
- **Offensive Impact (40%)**: Points, Assists, Shooting Efficiency, Usage Rate
- **Advanced Metrics (30%)**: TS%, eFG%, Offensive Rating, Game Score
- **Overall Impact (20%)**: Minutes Played, Rebounding
- **Efficiency (10%)**: Turnover management

### All-Finals Team Selection
Balanced evaluation including:
- **Offensive Contribution (35%)**: Scoring, playmaking, efficiency
- **Defensive Contribution (25%)**: Steals, blocks, defensive rating
- **Rebounding (15%)**: Total rebounds and rebounding percentage
- **Advanced Metrics (15%)**: Game Score, effective field goal percentage
- **Consistency (10%)**: Minutes played, turnover control

### All-Defensive Team Selection
Focused on defensive excellence:
- **Basic Defense (40%)**: Steals, blocks, defensive rebounds
- **Advanced Defense (35%)**: Defensive rating, steal%, block%, defensive rebound%
- **Rebounding Defense (15%)**: Total rebound% and limiting offensive rebounds
- **Consistency (10%)**: Minutes played, foul management

## CSV Output Files

### Detailed Analysis Files
- `NBA_Finals_2025_MVP_Analysis.csv` - Complete MVP scoring breakdown
- `NBA_Finals_2025_All_Finals_Analysis.csv` - All-Finals team scoring
- `NBA_Finals_2025_All_Defensive_Analysis.csv` - Defensive team scoring
- `NBA_Finals_2025_Complete_Stats.csv` - All player stats with calculated scores
- `NBA_Finals_2025_Team_Comparison.csv` - Team-by-team statistical comparison
- `NBA_Finals_2025_Statistical_Leaders.csv` - Category leaders

### Award Summary Files
- `NBA_Finals_2025_MVP_Winner.csv` - MVP winner details
- `NBA_Finals_2025_All_Finals_Team.csv` - Top 5 All-Finals selections
- `NBA_Finals_2025_All_Defensive_Team.csv` - Top 5 defensive selections

## Requirements

Install required packages:
```bash
pip install pandas numpy matplotlib seaborn
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## Statistical Methodology

The tool uses advanced statistical normalization and weighted scoring to ensure fair comparison across different statistical categories. All scores are normalized to a 0-100 scale before applying weights, ensuring no single category dominates the analysis.

### Key Advanced Stats Used
- **True Shooting %**: Overall shooting efficiency
- **Effective Field Goal %**: Shooting efficiency accounting for 3-pointers
- **Offensive/Defensive Rating**: Points per 100 possessions
- **Usage Rate**: Percentage of team plays used while on court
- **Game Score**: Comprehensive performance metric
- **Various percentage stats**: Steal%, Block%, Rebound%, etc.

## Features

✅ Deep statistical analysis across all major categories
✅ Position-aware team selection
✅ Advanced metrics integration
✅ Comprehensive CSV export functionality
✅ Interactive menu system
✅ Visualization generation
✅ Detailed written analysis and explanations
✅ Team comparison metrics
✅ Statistical leaders identification

## Notes

- Players with less than 50 total minutes are filtered out to focus on meaningful contributors
- All percentage and rate statistics are properly weighted by playing time
- Position classification is based on statistical profiles rather than listed positions
- Defensive metrics are weighted to favor actual defensive impact over traditional stats
