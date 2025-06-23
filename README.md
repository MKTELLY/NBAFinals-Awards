# NBA Finals 2025 Analysis Tool
 
A comprehensive statistical analysis tool for the NBA Finals 2025, determining MVP, All-Finals Team, and All-Defensive Team selections through advanced metrics and data visualization.

## Project Structure

```
NBA_Finals_2025_Project/
├── FINALS_2025_TOOL.py          # Main analysis script
├── FINALS_STATS_2025 - sportsref_download.xls.csv  # Raw Finals data
├── requirements.txt             # Python dependencies
├── README.md                   # This file
└── Output Files/
    ├── NBA_Finals_2025_Analysis_Report.txt
    ├── NBA_Finals_2025_Awards.csv
    └── Visualization PNGs/
```

## Installation & Setup

1. **Install Python Dependencies**:
```bash
pip install -r requirements.txt
```

Required packages:
- pandas (≥2.0.0)
- numpy (≥1.24.0)
- matplotlib (≥3.7.0)
- seaborn (≥0.12.0)

2. **Run the Analysis**:
```bash
python FINALS_2025_TOOL.py
```

## How It Works

The tool automatically loads Finals statistics data and performs comprehensive analysis using the `NBAFinalsAnalyzer` class. The analysis includes statistical normalization, weighted scoring algorithms, and position-based team selection.

## Core Features

### Award Determinations

#### Finals MVP Analysis
- **Weighted Scoring Algorithm** (0-100 scale):
  - **Offensive Impact (40%)**: Points, Assists, Shooting Efficiency, Usage Rate
  - **Advanced Metrics (30%)**: TS%, eFG%, Offensive Rating, Game Score
  - **Overall Impact (20%)**: Minutes Played, Rebounding
  - **Efficiency (10%)**: Turnover management

#### All-Finals Team Selection
- **Balanced Performance Evaluation**:
  - **Offensive Contribution (35%)**: Scoring, playmaking, efficiency
  - **Defensive Contribution (25%)**: Steals, blocks, defensive rating
  - **Rebounding (15%)**: Total rebounds and rebounding percentage
  - **Advanced Metrics (15%)**: Game Score, effective field goal percentage
  - **Consistency (10%)**: Minutes played, turnover control
- **Position-Aware Selection**: Attempts to balance guards, forwards, and centers

#### All-Defensive Team Selection
- **Defense-Focused Analysis**:
  - **Basic Defense (40%)**: Steals, blocks, defensive rebounds
  - **Advanced Defense (35%)**: Defensive rating, steal%, block%, defensive rebound%
  - **Rebounding Defense (15%)**: Total rebound% and limiting offensive rebounds
  - **Consistency (10%)**: Minutes played, foul management

### Data Visualization

The tool generates five comprehensive visualization sets:

1. **MVP Analysis** (`NBA_Finals_2025_MVP_Analysis.png`)
   - MVP score comparison for top 5 candidates
   - Scoring vs MVP score correlation
   - Component score breakdown for MVP winner
   - Usage rate vs shooting efficiency analysis

2. **Team Comparison** (`NBA_Finals_2025_Team_Comparison.png`)
   - Basic statistics comparison (PPG, RPG, APG, SPG, BPG)
   - Offensive efficiency metrics (TS%, ORtg)
   - Player distribution and defensive rating comparison

3. **Awards Dashboard** (`NBA_Finals_2025_Awards_Dashboard.png`)
   - MVP winner showcase with key statistics
   - All-Finals Team rankings
   - All-Defensive Team standings
   - Total awards distribution by team

4. **Statistical Leaders** (`NBA_Finals_2025_Statistical_Leaders.png`)
   - Top 5 players in six categories: Scoring, Rebounding, Assists, Steals, Blocks, Shooting Efficiency

5. **Defensive Analysis** (`NBA_Finals_2025_Defensive_Analysis.png`)
   - Defensive rating vs steals correlation
   - Blocks vs rebounds analysis
   - All-Defensive Team comparison
   - Team defensive metrics comparison

### Data Export

#### Single CSV Output
- **`NBA_Finals_2025_Awards.csv`** - Comprehensive awards file containing:
  - Finals MVP winner with key statistics
  - All-Finals Team (5 players) with rankings and performance metrics
  - All-Defensive Team (5 players) with defensive statistics
  - Award-specific analysis for each recipient

#### Generated Reports
- **`NBA_Finals_2025_Analysis_Report.txt`** - Complete written analysis with detailed explanations

## Interactive Menu Options

After running the script, you'll see a comprehensive analysis report followed by an interactive menu:

```
INDIVIDUAL ANALYSIS OPTIONS:
1. Get detailed MVP analysis
2. Get All-Finals Team breakdown  
3. Get All-Defensive Team breakdown
4. Get statistical leaders
5. Export data to CSV (run again)
6. Create visualizations
```

### Menu Functions:
- **Option 1**: Displays MVP winner with score breakdown and detailed analysis
- **Option 2**: Shows All-Finals Team rankings with composite scores
- **Option 3**: Lists All-Defensive Team with defensive metrics
- **Option 4**: Displays statistical leaders in major categories
- **Option 5**: Re-exports the awards CSV file
- **Option 6**: Generates all visualization charts

## Technical Implementation

### Data Processing
The `NBAFinalsAnalyzer` class handles:
- **Data Loading**: Parses CSV file with team-specific sections
- **Statistical Conversion**: Converts string data to numeric with proper handling
- **Player Filtering**: Removes players with <50 total minutes
- **Data Merging**: Combines regular and advanced statistics

### Statistical Methodology
- **Normalization**: All statistics normalized to 0-100 scale for fair comparison
- **Weighted Scoring**: Category-specific weights applied based on award type
- **Position Classification**: Automatic position assignment based on statistical profile:
  - **Guards**: High assists (≥4.0 APG) or assist percentage (≥25%)
  - **Centers**: High blocks (≥1.0 BPG) or rebounds (≥8.0 RPG + ≥3.0 BLK%)
  - **Forwards**: All other players

### Advanced Metrics Used
- **True Shooting %**: Overall shooting efficiency including free throws
- **Effective Field Goal %**: Shooting efficiency weighted for 3-pointers
- **Offensive/Defensive Rating**: Points per 100 possessions
- **Usage Rate**: Percentage of team plays used while on court
- **Game Score**: John Hollinger's comprehensive performance metric
- **Advanced Percentages**: Steal%, Block%, Rebound%, Turnover%, Assist%

## Visualization Features

All charts include:
- **Team Color Coding**: OKC Thunder (Blue) vs Indiana Pacers (Orange)
- **High-Resolution Output**: 300 DPI PNG files
- **Professional Styling**: Clean layouts with proper spacing and labels
- **Interactive Display**: Charts shown on screen before saving
- **Value Labels**: Statistical values displayed on bars and data points

## Sample Output

### Finals MVP Winner
```
Winner: [Player Name] ([Team])
MVP Score: [Score]/100
Key Stats: [PPG] PPG, [RPG] RPG, [APG] APG, [TS%] TS%
Analysis: [Detailed explanation of MVP selection]
```

### Awards CSV Structure
```csv
Award_Type,Rank,Player,Team,Position,Score,PPG,RPG,APG,SPG,BPG,TS%,ORtg,DRtg,Games,Analysis
Finals MVP,1,[Player],[Team],[Position],[Score],...
All-Finals Team,1,[Player],[Team],[Position],[Score],...
All-Defensive Team,1,[Player],[Team],[Position],[Score],...
```

## Key Features Summary

✅ **Comprehensive Statistical Analysis** - Advanced metrics across all major categories  
✅ **Multiple Award Determinations** - MVP, All-Finals Team, All-Defensive Team  
✅ **Professional Visualizations** - 5 different chart sets with team color coding  
✅ **Single CSV Export** - Award-focused output format  
✅ **Interactive Menu System** - 6 analysis options with detailed breakdowns  
✅ **Position-Aware Selection** - Balanced team composition  
✅ **Advanced Metrics Integration** - TS%, eFG%, ORtg, DRtg, Usage%, Game Score  
✅ **Team Comparison Analysis** - Head-to-head statistical breakdowns  
✅ **Statistical Leaders Identification** - Top performers in each category  
✅ **High-Quality Output** - Professional reports and visualizations  

## Important Notes

- **Minimum Playing Time**: Players with <50 total minutes filtered out for meaningful analysis
- **Statistical Weighting**: All metrics properly weighted by playing time and normalized (0-100 scale)
- **Position Classification**: Based on statistical profiles rather than listed positions
- **Team Color Scheme**: Oklahoma City Thunder (Blue), Indiana Pacers (Orange)
- **Data Source**: Official NBA Finals 2025 statistics from Basketball Reference
- **Export Format**: Award-centric CSV with comprehensive statistical breakdowns

---

**Created**: June 2025 | **NBA Finals Analysis Tool** | **Statistical Deep Dive**
