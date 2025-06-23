import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

class NBAFinalsAnalyzer:
    """
    Comprehensive NBA Finals 2025 Analysis Tool
    
    This tool analyzes Finals statistics to determine:
    1. Finals MVP based on weighted statistical analysis
    2. All-Finals Team (5 best overall players)
    3. Finals All-Defensive Team (5 best defensive players)
    """
    
    def __init__(self, csv_file_path: str):
        """Initialize the analyzer with Finals statistics data"""
        self.csv_file_path = csv_file_path
        self.players_df = None
        self.advanced_df = None
        self.load_data()
        
    def load_data(self):
        """Load and process the Finals statistics data"""
        try:
            # Read the CSV file
            with open(self.csv_file_path, 'r') as file:
                lines = file.readlines()
            
            # Find the data sections
            okc_start = None
            okc_advanced_start = None
            pacers_start = None
            pacers_advanced_start = None
            
            for i, line in enumerate(lines):
                if 'OKLAHOMA CITY THUNDER' in line:
                    okc_start = i + 1  # Header is next line
                elif 'INDIANA PACERS' in line:
                    pacers_start = i + 1
                elif 'Advanced' in line and okc_advanced_start is None:
                    okc_advanced_start = i + 1
                elif 'Advanced' in line and okc_advanced_start is not None:
                    pacers_advanced_start = i + 1
            
            # Extract regular stats
            okc_regular = self._extract_team_data(lines, okc_start, 'OKC')
            pacers_regular = self._extract_team_data(lines, pacers_start, 'IND')
            
            # Extract advanced stats
            okc_advanced = self._extract_advanced_data(lines, okc_advanced_start, 'OKC')
            pacers_advanced = self._extract_advanced_data(lines, pacers_advanced_start, 'IND')
            
            # Combine data
            all_regular = okc_regular + pacers_regular
            all_advanced = okc_advanced + pacers_advanced
            
            # Create DataFrames
            regular_columns = ['Rk', 'Player', 'Age', 'G', 'MP', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 
                             'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 
                             'FG%', '3P%', 'FT%', 'MP_PG', 'PTS_PG', 'TRB_PG', 'AST_PG', 'STL_PG', 'BLK_PG', 'Team']
            
            advanced_columns = ['Rk', 'Player', 'Age', 'G', 'GS', 'MP', 'TS%', 'eFG%', 'ORB%', 'DRB%', 
                               'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'ORtg', 'DRtg', 'GmSc', 'Team']
            
            self.players_df = pd.DataFrame(all_regular, columns=regular_columns)
            self.advanced_df = pd.DataFrame(all_advanced, columns=advanced_columns)
            
            # Convert numeric columns
            self._convert_numeric_columns()
            
            # Merge datasets
            self.combined_df = pd.merge(self.players_df, self.advanced_df, 
                                      on=['Player', 'Team'], suffixes=('', '_adv'))
            
            # Filter out players with minimal minutes (less than 50 total minutes)
            self.combined_df = self.combined_df[self.combined_df['MP'] >= 50].reset_index(drop=True)
            
            print(f"Successfully loaded data for {len(self.combined_df)} players")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def _extract_team_data(self, lines: List[str], start_idx: int, team: str) -> List[List]:
        """Extract regular statistics for a team"""
        data = []
        for i in range(start_idx + 1, len(lines)):  # Skip header
            line = lines[i].strip()
            if not line or line.startswith(',') or 'Totals' in line or 'Advanced' in line:
                break
            parts = line.split(',')
            if len(parts) >= 29 and parts[1]:  # Valid player row
                data.append(parts[:29] + [team])
        return data
    
    def _extract_advanced_data(self, lines: List[str], start_idx: int, team: str) -> List[List]:
        """Extract advanced statistics for a team"""
        data = []
        for i in range(start_idx + 1, len(lines)):  # Skip header
            line = lines[i].strip()
            if not line or line.startswith(','):
                break
            parts = line.split(',')
            if len(parts) >= 19 and parts[1]:  # Valid player row
                data.append(parts[:19] + [team])
        return data
    
    def _convert_numeric_columns(self):
        """Convert string columns to numeric where appropriate"""
        numeric_cols_regular = ['Age', 'G', 'MP', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 
                               'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
        
        numeric_cols_advanced = ['Age', 'G', 'GS', 'MP', 'ORtg', 'DRtg', 'GmSc']
        
        percentage_cols_regular = ['FG%', '3P%', 'FT%', 'MP_PG', 'PTS_PG', 'TRB_PG', 'AST_PG', 'STL_PG', 'BLK_PG']
        percentage_cols_advanced = ['TS%', 'eFG%', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%']
        
        # Convert regular stats
        for col in numeric_cols_regular:
            if col in self.players_df.columns:
                self.players_df[col] = pd.to_numeric(self.players_df[col], errors='coerce').fillna(0)
        
        for col in percentage_cols_regular:
            if col in self.players_df.columns:
                self.players_df[col] = pd.to_numeric(self.players_df[col], errors='coerce').fillna(0)
        
        # Convert advanced stats
        for col in numeric_cols_advanced:
            if col in self.advanced_df.columns:
                self.advanced_df[col] = pd.to_numeric(self.advanced_df[col], errors='coerce').fillna(0)
        
        for col in percentage_cols_advanced:
            if col in self.advanced_df.columns:
                self.advanced_df[col] = pd.to_numeric(self.advanced_df[col], errors='coerce').fillna(0)
    
    def calculate_mvp_score(self) -> pd.DataFrame:
        """
        Calculate MVP score based on weighted combination of stats
        
        MVP Criteria:
        - Offensive Impact: Points, Assists, Shooting Efficiency, Usage Rate
        - Advanced Metrics: TS%, eFG%, ORtg, GmSc
        - Overall Impact: Minutes Played, Game Score
        - Team Success: Factoring in team performance
        """
        df = self.combined_df.copy()
        
        # Normalize stats (0-100 scale)
        def normalize_stat(series, higher_is_better=True):
            if series.max() == series.min():
                return pd.Series([50] * len(series))
            if higher_is_better:
                return ((series - series.min()) / (series.max() - series.min())) * 100
            else:
                return ((series.max() - series) / (series.max() - series.min())) * 100
        
        # Offensive Impact (40% weight)
        df['PTS_norm'] = normalize_stat(df['PTS_PG'])
        df['AST_norm'] = normalize_stat(df['AST_PG'])
        df['TS_norm'] = normalize_stat(df['TS%'])
        df['USG_norm'] = normalize_stat(df['USG%'])
        df['offensive_score'] = (df['PTS_norm'] * 0.4 + df['AST_norm'] * 0.2 + 
                                df['TS_norm'] * 0.25 + df['USG_norm'] * 0.15)
        
        # Advanced Impact (30% weight)
        df['eFG_norm'] = normalize_stat(df['eFG%'])
        df['ORtg_norm'] = normalize_stat(df['ORtg'])
        df['GmSc_norm'] = normalize_stat(df['GmSc'])
        df['advanced_score'] = (df['eFG_norm'] * 0.3 + df['ORtg_norm'] * 0.35 + df['GmSc_norm'] * 0.35)
        
        # Overall Impact (20% weight)
        df['MP_norm'] = normalize_stat(df['MP_PG'])
        df['TRB_norm'] = normalize_stat(df['TRB_PG'])
        df['overall_score'] = (df['MP_norm'] * 0.6 + df['TRB_norm'] * 0.4)
        
        # Efficiency & Mistakes (10% weight)
        df['TOV_norm'] = normalize_stat(df['TOV'], higher_is_better=False)
        df['efficiency_score'] = df['TOV_norm']
        
        # Final MVP Score
        df['mvp_score'] = (df['offensive_score'] * 0.4 + 
                          df['advanced_score'] * 0.3 + 
                          df['overall_score'] * 0.2 + 
                          df['efficiency_score'] * 0.1)
        
        return df.sort_values('mvp_score', ascending=False)
    
    def calculate_all_finals_score(self) -> pd.DataFrame:
        """
        Calculate All-Finals Team score based on overall excellence
        
        Criteria:
        - Balanced offensive and defensive contribution
        - Advanced metrics
        - Consistency and efficiency
        """
        df = self.combined_df.copy()
        
        def normalize_stat(series, higher_is_better=True):
            if series.max() == series.min():
                return pd.Series([50] * len(series))
            if higher_is_better:
                return ((series - series.min()) / (series.max() - series.min())) * 100
            else:
                return ((series.max() - series) / (series.max() - series.min())) * 100
        
        # Offensive contribution (35%)
        df['PTS_norm'] = normalize_stat(df['PTS_PG'])
        df['AST_norm'] = normalize_stat(df['AST_PG'])
        df['TS_norm'] = normalize_stat(df['TS%'])
        df['offensive_contrib'] = (df['PTS_norm'] * 0.5 + df['AST_norm'] * 0.3 + df['TS_norm'] * 0.2)
        
        # Defensive contribution (25%)
        df['STL_norm'] = normalize_stat(df['STL_PG'])
        df['BLK_norm'] = normalize_stat(df['BLK_PG'])
        df['DRB_norm'] = normalize_stat(df['DRB'])
        df['DRtg_norm'] = normalize_stat(df['DRtg'], higher_is_better=False)
        df['defensive_contrib'] = (df['STL_norm'] * 0.25 + df['BLK_norm'] * 0.25 + 
                                  df['DRB_norm'] * 0.25 + df['DRtg_norm'] * 0.25)
        
        # Rebounding (15%)
        df['TRB_norm'] = normalize_stat(df['TRB_PG'])
        df['TRB_pct_norm'] = normalize_stat(df['TRB%'])
        df['rebounding_contrib'] = (df['TRB_norm'] * 0.6 + df['TRB_pct_norm'] * 0.4)
        
        # Advanced metrics (15%)
        df['GmSc_norm'] = normalize_stat(df['GmSc'])
        df['eFG_norm'] = normalize_stat(df['eFG%'])
        df['advanced_contrib'] = (df['GmSc_norm'] * 0.6 + df['eFG_norm'] * 0.4)
        
        # Consistency (10%)
        df['MP_norm'] = normalize_stat(df['MP_PG'])
        df['TOV_norm'] = normalize_stat(df['TOV'], higher_is_better=False)
        df['consistency_contrib'] = (df['MP_norm'] * 0.6 + df['TOV_norm'] * 0.4)
        
        # Final All-Finals Score
        df['all_finals_score'] = (df['offensive_contrib'] * 0.35 + 
                                 df['defensive_contrib'] * 0.25 + 
                                 df['rebounding_contrib'] * 0.15 + 
                                 df['advanced_contrib'] * 0.15 + 
                                 df['consistency_contrib'] * 0.10)
        
        return df.sort_values('all_finals_score', ascending=False)
    
    def calculate_defensive_score(self) -> pd.DataFrame:
        """
        Calculate All-Defensive Team score based on defensive metrics
        
        Criteria:
        - Defensive stats: Steals, Blocks, Defensive Rebounds
        - Advanced defensive metrics: DRtg, STL%, BLK%, DRB%
        - Defensive impact and consistency
        """
        df = self.combined_df.copy()
        
        def normalize_stat(series, higher_is_better=True):
            if series.max() == series.min():
                return pd.Series([50] * len(series))
            if higher_is_better:
                return ((series - series.min()) / (series.max() - series.min())) * 100
            else:
                return ((series.max() - series) / (series.max() - series.min())) * 100
        
        # Basic defensive stats (40%)
        df['STL_norm'] = normalize_stat(df['STL_PG'])
        df['BLK_norm'] = normalize_stat(df['BLK_PG'])
        df['DRB_norm'] = normalize_stat(df['DRB'])
        df['basic_defense'] = (df['STL_norm'] * 0.35 + df['BLK_norm'] * 0.35 + df['DRB_norm'] * 0.30)
        
        # Advanced defensive stats (35%)
        df['DRtg_norm'] = normalize_stat(df['DRtg'], higher_is_better=False)
        df['STL_pct_norm'] = normalize_stat(df['STL%'])
        df['BLK_pct_norm'] = normalize_stat(df['BLK%'])
        df['DRB_pct_norm'] = normalize_stat(df['DRB%'])
        df['advanced_defense'] = (df['DRtg_norm'] * 0.4 + df['STL_pct_norm'] * 0.2 + 
                                 df['BLK_pct_norm'] * 0.2 + df['DRB_pct_norm'] * 0.2)
        
        # Defensive rebounding impact (15%)
        df['TRB_pct_norm'] = normalize_stat(df['TRB%'])
        df['ORB_def_norm'] = normalize_stat(df['ORB'], higher_is_better=False)  # Preventing offensive rebounds
        df['rebounding_defense'] = (df['TRB_pct_norm'] * 0.7 + df['ORB_def_norm'] * 0.3)
        
        # Consistency and minutes (10%)
        df['MP_norm'] = normalize_stat(df['MP_PG'])
        df['PF_norm'] = normalize_stat(df['PF'], higher_is_better=False)  # Fewer fouls is better
        df['defensive_consistency'] = (df['MP_norm'] * 0.6 + df['PF_norm'] * 0.4)
        
        # Final Defensive Score
        df['defensive_score'] = (df['basic_defense'] * 0.40 + 
                               df['advanced_defense'] * 0.35 + 
                               df['rebounding_defense'] * 0.15 + 
                               df['defensive_consistency'] * 0.10)
        
        return df.sort_values('defensive_score', ascending=False)
    
    def get_finals_mvp(self) -> Dict[str, Any]:
        """Determine the Finals MVP"""
        mvp_df = self.calculate_mvp_score()
        mvp = mvp_df.iloc[0]
        
        return {
            'player': mvp['Player'],
            'team': mvp['Team'],
            'mvp_score': round(mvp['mvp_score'], 2),
            'key_stats': {
                'PPG': round(mvp['PTS_PG'], 1),
                'RPG': round(mvp['TRB_PG'], 1),
                'APG': round(mvp['AST_PG'], 1),
                'TS%': round(mvp['TS%'], 3),
                'ORtg': round(mvp['ORtg'], 1),
                'Games': mvp['G']
            },
            'analysis': self._generate_mvp_analysis(mvp)
        }
    
    def get_all_finals_team(self) -> List[Dict[str, Any]]:
        """Select the All-Finals Team (5 players)"""
        all_finals_df = self.calculate_all_finals_score()
        
        # Try to get positional balance
        team = []
        guards = []
        forwards = []
        centers = []
        
        # Classify players by position based on stats
        for _, player in all_finals_df.iterrows():
            if player['AST_PG'] >= 3.0 or player['AST%'] >= 20:  # Likely guard
                guards.append(player)
            elif player['BLK_PG'] >= 1.0 or player['TRB_PG'] >= 7.0:  # Likely center
                centers.append(player)
            else:  # Likely forward
                forwards.append(player)
        
        # Select balanced team
        team.extend(guards[:2])  # 2 guards
        team.extend(forwards[:2])  # 2 forwards
        team.extend(centers[:1])  # 1 center
        
        # Fill remaining spots with best available
        remaining_spots = 5 - len(team)
        used_players = [p['Player'] for p in team]
        
        for _, player in all_finals_df.iterrows():
            if len(team) >= 5:
                break
            if player['Player'] not in used_players:
                team.append(player)
        
        return [{
            'player': p['Player'],
            'team': p['Team'],
            'score': round(p['all_finals_score'], 2),
            'position': self._classify_position(p),
            'key_stats': {
                'PPG': round(p['PTS_PG'], 1),
                'RPG': round(p['TRB_PG'], 1),
                'APG': round(p['AST_PG'], 1),
                'SPG': round(p['STL_PG'], 1),
                'BPG': round(p['BLK_PG'], 1),
                'TS%': round(p['TS%'], 3)
            }
        } for p in team[:5]]
    
    def get_all_defensive_team(self) -> List[Dict[str, Any]]:
        """Select the All-Defensive Team (5 players)"""
        defensive_df = self.calculate_defensive_score()
        
        team = []
        for i in range(5):
            player = defensive_df.iloc[i]
            team.append({
                'player': player['Player'],
                'team': player['Team'],
                'defensive_score': round(player['defensive_score'], 2),
                'position': self._classify_position(player),
                'key_defensive_stats': {
                    'SPG': round(player['STL_PG'], 1),
                    'BPG': round(player['BLK_PG'], 1),
                    'DRB': round(player['DRB'], 1),
                    'DRtg': round(player['DRtg'], 1),
                    'STL%': round(player['STL%'], 1),
                    'BLK%': round(player['BLK%'], 1),
                    'DRB%': round(player['DRB%'], 1)
                }
            })
        
        return team
    
    def _classify_position(self, player) -> str:
        """Classify player position based on stats"""
        if player['AST_PG'] >= 4.0 or player['AST%'] >= 25:
            return "Guard"
        elif player['BLK_PG'] >= 1.0 or (player['TRB_PG'] >= 8.0 and player['BLK%'] >= 3.0):
            return "Center"
        else:
            return "Forward"
    
    def _generate_mvp_analysis(self, mvp) -> str:
        """Generate detailed MVP analysis"""
        analysis = f"{mvp['Player']} dominates the Finals MVP race with exceptional "
        
        if mvp['PTS_PG'] >= 25:
            analysis += f"scoring ({mvp['PTS_PG']:.1f} PPG), "
        if mvp['AST_PG'] >= 5:
            analysis += f"playmaking ({mvp['AST_PG']:.1f} APG), "
        if mvp['TS%'] >= 0.55:
            analysis += f"elite efficiency ({mvp['TS%']:.1%} TS%), "
        if mvp['ORtg'] >= 115:
            analysis += f"outstanding offensive rating ({mvp['ORtg']:.1f}), "
        
        analysis += f"leading {mvp['Team']} with consistent excellence across {mvp['G']} games."
        
        return analysis
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive Finals analysis report"""
        mvp = self.get_finals_mvp()
        all_finals = self.get_all_finals_team()
        all_defensive = self.get_all_defensive_team()
        
        report = "="*80 + "\n"
        report += "NBA FINALS 2025 - COMPREHENSIVE STATISTICAL ANALYSIS\n"
        report += "="*80 + "\n\n"
        
        # Finals MVP
        report += "FINALS MVP\n"
        report += "-"*40 + "\n"
        report += f"Winner: {mvp['player']} ({mvp['team']})\n"
        report += f"MVP Score: {mvp['mvp_score']}/100\n"
        report += f"Key Stats: {mvp['key_stats']['PPG']} PPG, {mvp['key_stats']['RPG']} RPG, "
        report += f"{mvp['key_stats']['APG']} APG, {mvp['key_stats']['TS%']:.1%} TS%\n"
        report += f"Analysis: {mvp['analysis']}\n\n"
        
        # All-Finals Team
        report += "ALL-FINALS TEAM\n"
        report += "-"*40 + "\n"
        for i, player in enumerate(all_finals, 1):
            report += f"{i}. {player['player']} ({player['team']}) - {player['position']}\n"
            report += f"   Score: {player['score']}/100\n"
            report += f"   Stats: {player['key_stats']['PPG']} PPG, {player['key_stats']['RPG']} RPG, "
            report += f"{player['key_stats']['APG']} APG, {player['key_stats']['SPG']} SPG, {player['key_stats']['BPG']} BPG\n\n"
        
        # All-Defensive Team
        report += "ALL-DEFENSIVE TEAM\n"
        report += "-"*40 + "\n"
        for i, player in enumerate(all_defensive, 1):
            report += f"{i}. {player['player']} ({player['team']}) - {player['position']}\n"
            report += f"   Defensive Score: {player['defensive_score']}/100\n"
            stats = player['key_defensive_stats']
            report += f"   Stats: {stats['SPG']} SPG, {stats['BPG']} BPG, {stats['DRB']} DRB, "
            report += f"{stats['DRtg']} DRtg, {stats['STL%']:.1f}% STL%, {stats['BLK%']:.1f}% BLK%\n\n"
        
        # Statistical Leaders
        report += "STATISTICAL LEADERS\n"
        report += "-"*40 + "\n"
        
        # Points leader
        pts_leader = self.combined_df.loc[self.combined_df['PTS_PG'].idxmax()]
        report += f"Scoring: {pts_leader['Player']} ({pts_leader['Team']}) - {pts_leader['PTS_PG']:.1f} PPG\n"
        
        # Rebounds leader
        reb_leader = self.combined_df.loc[self.combined_df['TRB_PG'].idxmax()]
        report += f"Rebounding: {reb_leader['Player']} ({reb_leader['Team']}) - {reb_leader['TRB_PG']:.1f} RPG\n"
        
        # Assists leader
        ast_leader = self.combined_df.loc[self.combined_df['AST_PG'].idxmax()]
        report += f"Assists: {ast_leader['Player']} ({ast_leader['Team']}) - {ast_leader['AST_PG']:.1f} APG\n"
        
        # Steals leader
        stl_leader = self.combined_df.loc[self.combined_df['STL_PG'].idxmax()]
        report += f"Steals: {stl_leader['Player']} ({stl_leader['Team']}) - {stl_leader['STL_PG']:.1f} SPG\n"
        
        # Blocks leader
        blk_leader = self.combined_df.loc[self.combined_df['BLK_PG'].idxmax()]
        report += f"Blocks: {blk_leader['Player']} ({blk_leader['Team']}) - {blk_leader['BLK_PG']:.1f} BPG\n"
        
        # Efficiency leaders
        ts_leader = self.combined_df.loc[self.combined_df['TS%'].idxmax()]
        report += f"True Shooting%: {ts_leader['Player']} ({ts_leader['Team']}) - {ts_leader['TS%']:.1%}\n"
        
        ortg_leader = self.combined_df.loc[self.combined_df['ORtg'].idxmax()]
        report += f"Offensive Rating: {ortg_leader['Player']} ({ortg_leader['Team']}) - {ortg_leader['ORtg']:.1f}\n"
        
        drtg_leader = self.combined_df.loc[self.combined_df['DRtg'].idxmin()]
        report += f"Defensive Rating: {drtg_leader['Player']} ({drtg_leader['Team']}) - {drtg_leader['DRtg']:.1f}\n"
        
        report += "\n" + "="*80 + "\n"
        report += "Analysis complete. All selections based on comprehensive statistical evaluation.\n"
        report += "="*80
        
        return report
    def export_to_csv(self):
        """Export award-based analysis to a single CSV file"""
        try:
            # Get award winners
            mvp = self.get_finals_mvp()
            all_finals_team = self.get_all_finals_team()
            defensive_team = self.get_all_defensive_team()
            
            # Create comprehensive awards data
            awards_data = []
            
            # Add MVP
            awards_data.append({
                'Award_Type': 'Finals MVP',
                'Rank': 1,
                'Player': mvp['player'],
                'Team': mvp['team'],
                'Position': self._classify_position_from_stats(mvp),
                'Score': mvp['mvp_score'],
                'PPG': mvp['key_stats']['PPG'],
                'RPG': mvp['key_stats']['RPG'],
                'APG': mvp['key_stats']['APG'],
                'SPG': None,  # Not in MVP stats
                'BPG': None,  # Not in MVP stats
                'TS%': mvp['key_stats']['TS%'],
                'ORtg': mvp['key_stats']['ORtg'],
                'DRtg': None,  # Not in MVP stats
                'Games': mvp['key_stats']['Games'],
                'Analysis': mvp['analysis']
            })
            
            # Add All-Finals Team
            for i, player in enumerate(all_finals_team):
                awards_data.append({
                    'Award_Type': 'All-Finals Team',
                    'Rank': i + 1,
                    'Player': player['player'],
                    'Team': player['team'],
                    'Position': player['position'],
                    'Score': player['score'],
                    'PPG': player['key_stats']['PPG'],
                    'RPG': player['key_stats']['RPG'],
                    'APG': player['key_stats']['APG'],
                    'SPG': player['key_stats']['SPG'],
                    'BPG': player['key_stats']['BPG'],
                    'TS%': player['key_stats']['TS%'],
                    'ORtg': None,  # Not in All-Finals stats
                    'DRtg': None,  # Not in All-Finals stats
                    'Games': None,  # Not in All-Finals stats
                    'Analysis': f"All-around excellence with {player['key_stats']['PPG']} PPG, {player['key_stats']['RPG']} RPG, {player['key_stats']['APG']} APG"
                })
            
            # Add All-Defensive Team
            for i, player in enumerate(defensive_team):
                awards_data.append({
                    'Award_Type': 'All-Defensive Team',
                    'Rank': i + 1,
                    'Player': player['player'],
                    'Team': player['team'],
                    'Position': player['position'],
                    'Score': player['defensive_score'],
                    'PPG': None,  # Not relevant for defensive award
                    'RPG': None,  # Will use DRB instead
                    'APG': None,  # Not relevant for defensive award
                    'SPG': player['key_defensive_stats']['SPG'],
                    'BPG': player['key_defensive_stats']['BPG'],
                    'TS%': None,  # Not relevant for defensive award
                    'ORtg': None,  # Not relevant for defensive award
                    'DRtg': player['key_defensive_stats']['DRtg'],
                    'Games': None,  # Not in defensive stats
                    'Analysis': f"Defensive anchor with {player['key_defensive_stats']['SPG']} SPG, {player['key_defensive_stats']['BPG']} BPG, {player['key_defensive_stats']['DRtg']} DRtg"
                })
            
            # Create DataFrame
            finals_awards_df = pd.DataFrame(awards_data)
            
            # Add additional defensive stats for All-Defensive Team entries
            for i, row in finals_awards_df.iterrows():
                if row['Award_Type'] == 'All-Defensive Team':
                    # Find the defensive player data
                    def_player = next(p for p in defensive_team if p['player'] == row['Player'])
                    finals_awards_df.at[i, 'DRB'] = def_player['key_defensive_stats']['DRB']
                    finals_awards_df.at[i, 'STL%'] = f"{def_player['key_defensive_stats']['STL%']:.1f}%"
                    finals_awards_df.at[i, 'BLK%'] = f"{def_player['key_defensive_stats']['BLK%']:.1f}%"
                    finals_awards_df.at[i, 'DRB%'] = f"{def_player['key_defensive_stats']['DRB%']:.1f}%"
            
            # Reorder columns for better presentation
            column_order = ['Award_Type', 'Rank', 'Player', 'Team', 'Position', 'Score', 'PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TS%', 'ORtg', 'DRtg', 'Games']
            other_columns = [col for col in finals_awards_df.columns if col not in column_order]
            finals_awards_df = finals_awards_df[column_order + other_columns]
            
            # Round numeric columns
            for col in ['Score', 'PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TS%', 'ORtg', 'DRtg', 'DRB']:
                if col in finals_awards_df.columns:
                    finals_awards_df[col] = pd.to_numeric(finals_awards_df[col], errors='ignore')
                    if finals_awards_df[col].dtype in ['float64', 'int64']:
                        finals_awards_df[col] = finals_awards_df[col].round(1)
            
            # Export to CSV
            finals_awards_df.to_csv('NBA_Finals_2025_Awards.csv', index=False)
            
            print("CSV Export Complete!")
            print("Single file created: NBA_Finals_2025_Awards.csv")
            print(f"Contains {len(finals_awards_df)} total award recipients:")
            print(f"  - 1 Finals MVP")
            print(f"  - 5 All-Finals Team members")
            print(f"  - 5 All-Defensive Team members")
            
            return True
            
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
    
    def _classify_position_from_stats(self, mvp_data):
        """Helper method to classify position from MVP data"""
        # Access the combined_df to get the player's full stats
        player_row = self.combined_df[self.combined_df['Player'] == mvp_data['player']].iloc[0]
        return self._classify_position(player_row)
    
    def create_visualizations(self):
        """Create comprehensive visualizations of the Finals analysis"""
        print("Creating NBA Finals 2025 visualizations...")
        
        # Set style for better looking plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create visualizations
        self._create_mvp_comparison()
        self._create_team_comparison()
        self._create_award_winners_dashboard()
        self._create_statistical_leaders()
        self._create_defensive_analysis()
        
        print("All visualizations saved successfully!")
        
    def _create_mvp_comparison(self):
        """Create MVP comparison visualization"""
        mvp_df = self.calculate_mvp_score()
        top_5_mvp = mvp_df.head(5)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('NBA Finals 2025 - MVP Race Analysis', fontsize=20, fontweight='bold', y=0.95)
        
        # Colors for teams
        colors = ['#1f77b4' if team == 'OKC' else '#ff7f0e' for team in top_5_mvp['Team']]
        
        # MVP Score comparison
        bars1 = ax1.bar(range(len(top_5_mvp)), top_5_mvp['mvp_score'], color=colors)
        ax1.set_title('MVP Score Comparison (Top 5)', fontweight='bold', pad=20)
        ax1.set_ylabel('MVP Score')
        ax1.set_xticks(range(len(top_5_mvp)))
        ax1.set_xticklabels([f"{row['Player']}\n({row['Team']})" for _, row in top_5_mvp.iterrows()], 
                           rotation=0, ha='center', fontsize=9)
        
        # Add score labels on bars
        for bar, score in zip(bars1, top_5_mvp['mvp_score']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # PPG vs MVP Score scatter
        ax2.scatter(top_5_mvp['PTS_PG'], top_5_mvp['mvp_score'], 
                   c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=2)
        ax2.set_xlabel('Points Per Game')
        ax2.set_ylabel('MVP Score')
        ax2.set_title('Scoring vs MVP Score', fontweight='bold', pad=20)
        for _, row in top_5_mvp.iterrows():
            ax2.annotate(row['Player'], (row['PTS_PG'], row['mvp_score']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Component scores breakdown for MVP winner
        mvp_winner = top_5_mvp.iloc[0]
        components = ['Offensive\nScore', 'Advanced\nScore', 'Overall\nScore', 'Efficiency\nScore']
        scores = [mvp_winner['offensive_score'], mvp_winner['advanced_score'], 
                 mvp_winner['overall_score'], mvp_winner['efficiency_score']]
        
        bars3 = ax3.bar(components, scores, color=['#2ca02c', '#d62728', '#9467bd', '#8c564b'])
        ax3.set_title(f'{mvp_winner["Player"]} - MVP Component Breakdown', fontweight='bold', pad=20)
        ax3.set_ylabel('Component Score')
        ax3.tick_params(axis='x', labelsize=9)
        for bar, score in zip(bars3, scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Efficiency comparison (TS% vs Usage%)
        ax4.scatter(top_5_mvp['USG%'], top_5_mvp['TS%'], 
                   c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=2)
        ax4.set_xlabel('Usage Rate (%)')
        ax4.set_ylabel('True Shooting %')
        ax4.set_title('Usage vs Efficiency', fontweight='bold', pad=20)
        for _, row in top_5_mvp.iterrows():
            ax4.annotate(row['Player'], (row['USG%'], row['TS%']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(top=0.9)
        plt.savefig('NBA_Finals_2025_MVP_Analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _create_team_comparison(self):
        """Create team comparison visualization"""
        team_stats = self.combined_df.groupby('Team').agg({
            'PTS_PG': 'mean',
            'TRB_PG': 'mean',
            'AST_PG': 'mean',
            'STL_PG': 'mean',
            'BLK_PG': 'mean',
            'TS%': 'mean',
            'ORtg': 'mean',
            'DRtg': 'mean'
        }).round(1)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('NBA Finals 2025 - Team Comparison', fontsize=20, fontweight='bold', y=0.95)
        
        teams = team_stats.index
        colors = ['#1f77b4', '#ff7f0e']  # OKC blue, IND orange
        
        # Basic stats comparison
        stats = ['PPG', 'RPG', 'APG', 'SPG', 'BPG']
        x = np.arange(len(stats))
        width = 0.35
        
        okc_stats = [team_stats.loc['OKC', 'PTS_PG'], team_stats.loc['OKC', 'TRB_PG'], 
                    team_stats.loc['OKC', 'AST_PG'], team_stats.loc['OKC', 'STL_PG'], 
                    team_stats.loc['OKC', 'BLK_PG']]
        ind_stats = [team_stats.loc['IND', 'PTS_PG'], team_stats.loc['IND', 'TRB_PG'], 
                    team_stats.loc['IND', 'AST_PG'], team_stats.loc['IND', 'STL_PG'], 
                    team_stats.loc['IND', 'BLK_PG']]
        
        bars1 = ax1.bar(x - width/2, okc_stats, width, label='Oklahoma City Thunder', color=colors[0])
        bars2 = ax1.bar(x + width/2, ind_stats, width, label='Indiana Pacers', color=colors[1])
        
        ax1.set_xlabel('Statistics')
        ax1.set_ylabel('Per Game Average')
        ax1.set_title('Basic Statistics Comparison', fontweight='bold', pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(stats)
        ax1.legend(loc='upper left')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Shooting efficiency
        eff_stats = ['TS%', 'ORtg']
        okc_eff = [team_stats.loc['OKC', 'TS%'], team_stats.loc['OKC', 'ORtg']]
        ind_eff = [team_stats.loc['IND', 'TS%'], team_stats.loc['IND', 'ORtg']]
        
        x2 = np.arange(len(eff_stats))
        bars3 = ax2.bar(x2 - width/2, okc_eff, width, label='Oklahoma City Thunder', color=colors[0])
        bars4 = ax2.bar(x2 + width/2, ind_eff, width, label='Indiana Pacers', color=colors[1])
        
        ax2.set_xlabel('Efficiency Metrics')
        ax2.set_ylabel('Value')
        ax2.set_title('Offensive Efficiency Comparison', fontweight='bold', pad=20)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(eff_stats)
        ax2.legend(loc='upper left')
        
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Player distribution by team
        team_players = self.combined_df['Team'].value_counts()
        wedges, texts, autotexts = ax3.pie(team_players.values, labels=team_players.index, 
                                          colors=colors, autopct='%1.0f players',
                                          startangle=90, textprops={'fontsize': 10})
        ax3.set_title('Player Distribution', fontweight='bold', pad=20)
        
        # Defensive rating comparison
        drtg_data = [team_stats.loc['OKC', 'DRtg'], team_stats.loc['IND', 'DRtg']]
        bars5 = ax4.bar(teams, drtg_data, color=colors, width=0.6)
        ax4.set_xlabel('Team')
        ax4.set_ylabel('Defensive Rating (Lower is Better)')
        ax4.set_title('Defensive Rating Comparison', fontweight='bold', pad=20)
        
        for bar, value in zip(bars5, drtg_data):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(top=0.9)
        plt.savefig('NBA_Finals_2025_Team_Comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _create_award_winners_dashboard(self):
        """Create award winners dashboard"""
        mvp = self.get_finals_mvp()
        all_finals_team = self.get_all_finals_team()
        defensive_team = self.get_all_defensive_team()
        
        fig = plt.figure(figsize=(22, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
        
        # Title
        fig.suptitle('NBA Finals 2025 - Award Winners Dashboard', fontsize=26, fontweight='bold', y=0.97)
        
        # MVP Section
        ax_mvp = fig.add_subplot(gs[0, :2])
        mvp_stats = [mvp['key_stats']['PPG'], mvp['key_stats']['RPG'], mvp['key_stats']['APG']]
        mvp_labels = ['PPG', 'RPG', 'APG']
        colors_mvp = ['#1f77b4' if mvp['team'] == 'OKC' else '#ff7f0e']
        
        bars_mvp = ax_mvp.bar(mvp_labels, mvp_stats, color=colors_mvp[0], alpha=0.8)
        ax_mvp.set_title(f'Finals MVP: {mvp["player"]} ({mvp["team"]})\nScore: {mvp["mvp_score"]}/100', 
                        fontweight='bold', fontsize=14)
        ax_mvp.set_ylabel('Per Game Stats')
        
        for bar, stat in zip(bars_mvp, mvp_stats):
            ax_mvp.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                       f'{stat:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # All-Finals Team
        ax_finals = fig.add_subplot(gs[0, 2:])
        finals_players = [p['player'] for p in all_finals_team]
        finals_scores = [p['score'] for p in all_finals_team]
        team_colors = ['#1f77b4' if p['team'] == 'OKC' else '#ff7f0e' for p in all_finals_team]
        
        bars_finals = ax_finals.barh(range(len(finals_players)), finals_scores, color=team_colors, alpha=0.8)
        ax_finals.set_title('All-Finals Team', fontweight='bold', fontsize=14)
        ax_finals.set_xlabel('All-Finals Score')
        ax_finals.set_yticks(range(len(finals_players)))
        ax_finals.set_yticklabels([f"{p['player']} ({p['team']})" for p in all_finals_team])
        ax_finals.invert_yaxis()
        
        for bar, score in zip(bars_finals, finals_scores):
            ax_finals.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                          f'{score:.1f}', ha='left', va='center', fontweight='bold')
        
        # Defensive Team
        ax_def = fig.add_subplot(gs[1, :2])
        def_players = [p['player'] for p in defensive_team]
        def_scores = [p['defensive_score'] for p in defensive_team]
        def_colors = ['#1f77b4' if p['team'] == 'OKC' else '#ff7f0e' for p in defensive_team]
        
        bars_def = ax_def.barh(range(len(def_players)), def_scores, color=def_colors, alpha=0.8)
        ax_def.set_title('All-Defensive Team', fontweight='bold', fontsize=14)
        ax_def.set_xlabel('Defensive Score')
        ax_def.set_yticks(range(len(def_players)))
        ax_def.set_yticklabels([f"{p['player']} ({p['team']})" for p in defensive_team])
        ax_def.invert_yaxis()
        
        for bar, score in zip(bars_def, def_scores):
            ax_def.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                       f'{score:.1f}', ha='left', va='center', fontweight='bold')
        
        # Statistical comparison of award winners
        ax_compare = fig.add_subplot(gs[1, 2:])
        
        # Get stats for comparison
        mvp_player_stats = self.combined_df[self.combined_df['Player'] == mvp['player']].iloc[0]
        all_finals_avg_ppg = np.mean([p['key_stats']['PPG'] for p in all_finals_team])
        def_avg_spg = np.mean([p['key_defensive_stats']['SPG'] for p in defensive_team])
        def_avg_bpg = np.mean([p['key_defensive_stats']['BPG'] for p in defensive_team])
        
        categories = ['MVP PPG', 'All-Finals\nAvg PPG', 'All-Def\nAvg SPG', 'All-Def\nAvg BPG']
        values = [mvp_player_stats['PTS_PG'], all_finals_avg_ppg, def_avg_spg, def_avg_bpg]
        colors_comp = ['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']
        
        bars_comp = ax_compare.bar(categories, values, color=colors_comp, alpha=0.8)
        ax_compare.set_title('Award Winners Statistical Highlights', fontweight='bold', fontsize=14)
        ax_compare.set_ylabel('Statistics')
        
        for bar, value in zip(bars_comp, values):
            ax_compare.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                           f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
          # Team representation in awards
        ax_team_rep = fig.add_subplot(gs[3, :])
        
        # Count awards by team
        okc_awards = 0
        ind_awards = 0
        
        if mvp['team'] == 'OKC':
            okc_awards += 1
        else:
            ind_awards += 1
            
        for player in all_finals_team:
            if player['team'] == 'OKC':
                okc_awards += 1
            else:
                ind_awards += 1
                
        for player in defensive_team:
            if player['team'] == 'OKC':
                okc_awards += 1
            else:
                ind_awards += 1
        
        team_awards = ['Oklahoma City Thunder', 'Indiana Pacers']
        award_counts = [okc_awards, ind_awards]
        
        bars_team = ax_team_rep.bar(team_awards, award_counts, color=['#1f77b4', '#ff7f0e'], 
                                   alpha=0.8, width=0.5)
        ax_team_rep.set_title('Total Awards by Team', fontweight='bold', fontsize=18, pad=25)
        ax_team_rep.set_ylabel('Number of Awards', fontsize=14)
        ax_team_rep.tick_params(axis='x', labelsize=12)
        ax_team_rep.tick_params(axis='y', labelsize=12)
        
        for bar, count in zip(bars_team, award_counts):
            ax_team_rep.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                            f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=16)
        
        # Add legend
        okc_patch = mpatches.Patch(color='#1f77b4', label='Oklahoma City Thunder')
        ind_patch = mpatches.Patch(color='#ff7f0e', label='Indiana Pacers')
        fig.legend(handles=[okc_patch, ind_patch], loc='upper right', 
                  bbox_to_anchor=(0.98, 0.95), fontsize=12)
        
        plt.savefig('NBA_Finals_2025_Awards_Dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _create_statistical_leaders(self):
        """Create statistical leaders visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('NBA Finals 2025 - Statistical Leaders', fontsize=22, fontweight='bold', y=0.95)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        stats = ['PTS_PG', 'TRB_PG', 'AST_PG', 'STL_PG', 'BLK_PG', 'TS%']
        titles = ['Scoring Leaders (PPG)', 'Rebounding Leaders (RPG)', 'Assists Leaders (APG)', 
                 'Steals Leaders (SPG)', 'Blocks Leaders (BPG)', 'Shooting Efficiency (TS%)']
        
        for i, (stat, title) in enumerate(zip(stats, titles)):
            # Get top 5 players for this stat
            top_players = self.combined_df.nlargest(5, stat)
            
            colors = ['#1f77b4' if team == 'OKC' else '#ff7f0e' for team in top_players['Team']]
            
            bars = axes[i].barh(range(len(top_players)), top_players[stat], color=colors, alpha=0.8)
            axes[i].set_title(title, fontweight='bold', pad=15, fontsize=14)
            axes[i].set_yticks(range(len(top_players)))
            axes[i].set_yticklabels([f"{row['Player']} ({row['Team']})" for _, row in top_players.iterrows()],
                                  fontsize=10)
            axes[i].invert_yaxis()
            axes[i].tick_params(axis='x', labelsize=10)
            
            # Add value labels
            for bar, value in zip(bars, top_players[stat]):
                label_format = f'{value:.1%}' if stat == 'TS%' else f'{value:.1f}'
                axes[i].text(bar.get_width() + (0.01 if stat == 'TS%' else 0.1), 
                           bar.get_y() + bar.get_height()/2,
                           label_format, ha='left', va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(top=0.9)
        plt.savefig('NBA_Finals_2025_Statistical_Leaders.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _create_defensive_analysis(self):
        """Create defensive analysis visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('NBA Finals 2025 - Defensive Analysis', fontsize=22, fontweight='bold', y=0.95)
        
        # Defensive rating vs steals
        colors = ['#1f77b4' if team == 'OKC' else '#ff7f0e' for team in self.combined_df['Team']]
        
        ax1.scatter(self.combined_df['DRtg'], self.combined_df['STL_PG'], 
                   c=colors, s=120, alpha=0.7, edgecolors='black')
        ax1.set_xlabel('Defensive Rating (Lower is Better)', fontsize=12)
        ax1.set_ylabel('Steals Per Game', fontsize=12)
        ax1.set_title('Defensive Rating vs Steals', fontweight='bold', pad=15, fontsize=14)
        ax1.invert_xaxis()  # Lower DRtg is better
        ax1.grid(True, alpha=0.3)
        
        # Blocks vs rebounds
        ax2.scatter(self.combined_df['BLK_PG'], self.combined_df['TRB_PG'], 
                   c=colors, s=120, alpha=0.7, edgecolors='black')
        ax2.set_xlabel('Blocks Per Game', fontsize=12)
        ax2.set_ylabel('Total Rebounds Per Game', fontsize=12)
        ax2.set_title('Blocks vs Rebounds', fontweight='bold', pad=15, fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Defensive team comparison
        defensive_team = self.get_all_defensive_team()
        def_players = [p['player'] for p in defensive_team]
        def_spg = [p['key_defensive_stats']['SPG'] for p in defensive_team]
        def_bpg = [p['key_defensive_stats']['BPG'] for p in defensive_team]
        def_colors = ['#1f77b4' if p['team'] == 'OKC' else '#ff7f0e' for p in defensive_team]
        
        x = np.arange(len(def_players))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, def_spg, width, label='Steals Per Game', 
                       color='lightblue', alpha=0.8, edgecolor='black')
        bars2 = ax3.bar(x + width/2, def_bpg, width, label='Blocks Per Game', 
                       color='orange', alpha=0.8, edgecolor='black')
        
        ax3.set_xlabel('All-Defensive Team Players', fontsize=12)
        ax3.set_ylabel('Per Game Stats', fontsize=12)
        ax3.set_title('All-Defensive Team - Steals vs Blocks', fontweight='bold', pad=15, fontsize=14)
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"{p}\n({defensive_team[i]['team']})" for i, p in enumerate(def_players)], 
                           rotation=0, ha='center', fontsize=9)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Team defensive comparison
        team_def_stats = self.combined_df.groupby('Team').agg({
            'STL_PG': 'mean',
            'BLK_PG': 'mean',
            'DRtg': 'mean',
            'DRB': 'mean'
        }).round(1)
        
        teams = team_def_stats.index
        categories = ['Steals PG', 'Blocks PG', 'Def Rebounds', 'Def Rating']
        
        okc_def = [team_def_stats.loc['OKC', 'STL_PG'], team_def_stats.loc['OKC', 'BLK_PG'], 
                  team_def_stats.loc['OKC', 'DRB'], team_def_stats.loc['OKC', 'DRtg']]
        ind_def = [team_def_stats.loc['IND', 'STL_PG'], team_def_stats.loc['IND', 'BLK_PG'], 
                  team_def_stats.loc['IND', 'DRB'], team_def_stats.loc['IND', 'DRtg']]
        
        x4 = np.arange(len(categories))
        width = 0.35
        
        bars3 = ax4.bar(x4 - width/2, okc_def, width, label='Oklahoma City Thunder', 
                       color='#1f77b4', alpha=0.8, edgecolor='black')
        bars4 = ax4.bar(x4 + width/2, ind_def, width, label='Indiana Pacers', 
                       color='#ff7f0e', alpha=0.8, edgecolor='black')
        
        ax4.set_xlabel('Defensive Categories', fontsize=12)
        ax4.set_ylabel('Average Values', fontsize=12)
        ax4.set_title('Team Defensive Comparison', fontweight='bold', pad=15, fontsize=14)
        ax4.set_xticks(x4)
        ax4.set_xticklabels(categories, fontsize=10)
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(top=0.9)
        plt.savefig('NBA_Finals_2025_Defensive_Analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main execution function"""
    print("NBA Finals 2025 Analysis Tool")
    print("=" * 50)
    
    # Initialize analyzer
    csv_file = "FINALS_STATS_2025 - sportsref_download.xls.csv"
    analyzer = NBAFinalsAnalyzer(csv_file)
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()
    print(report)
    
    # Save detailed analysis to file
    with open('NBA_Finals_2025_Analysis_Report.txt', 'w') as f:
        f.write(report)
    
    print("\nDetailed report saved as 'NBA_Finals_2025_Analysis_Report.txt'")
    
    # Export to CSV
    print("\nExporting awards data to single CSV file...")
    analyzer.export_to_csv()
    
    # Individual analysis options
    print("\n" + "="*50)
    print("INDIVIDUAL ANALYSIS OPTIONS:")
    print("1. Get detailed MVP analysis")
    print("2. Get All-Finals Team breakdown")
    print("3. Get All-Defensive Team breakdown")
    print("4. Get statistical leaders")
    print("5. Export data to CSV (run again)")
    print("6. Create visualizations")
    
    while True:
        choice = input("\nEnter choice (1-6) or 'quit' to exit: ").strip()
        
        if choice.lower() == 'quit':
            break
        elif choice == '1':
            mvp = analyzer.get_finals_mvp()
            print(f"\nMVP: {mvp['player']} ({mvp['team']})")
            print(f"Score: {mvp['mvp_score']}/100")
            print(f"Analysis: {mvp['analysis']}")
        elif choice == '2':
            team = analyzer.get_all_finals_team()
            print("\nALL-FINALS TEAM:")
            for i, player in enumerate(team, 1):
                print(f"{i}. {player['player']} ({player['team']}) - {player['position']}")
                print(f"   Score: {player['score']}/100")
        elif choice == '3':
            def_team = analyzer.get_all_defensive_team()
            print("\nALL-DEFENSIVE TEAM:")
            for i, player in enumerate(def_team, 1):
                print(f"{i}. {player['player']} ({player['team']}) - {player['position']}")
                print(f"   Defensive Score: {player['defensive_score']}/100")
        elif choice == '4':
            df = analyzer.combined_df
            print("\nSTATISTICAL LEADERS:")
            print(f"Points: {df.loc[df['PTS_PG'].idxmax(), 'Player']} - {df['PTS_PG'].max():.1f} PPG")
            print(f"Rebounds: {df.loc[df['TRB_PG'].idxmax(), 'Player']} - {df['TRB_PG'].max():.1f} RPG")
            print(f"Assists: {df.loc[df['AST_PG'].idxmax(), 'Player']} - {df['AST_PG'].max():.1f} APG")
            print(f"Steals: {df.loc[df['STL_PG'].idxmax(), 'Player']} - {df['STL_PG'].max():.1f} SPG")
            print(f"Blocks: {df.loc[df['BLK_PG'].idxmax(), 'Player']} - {df['BLK_PG'].max():.1f} BPG")
        elif choice == '5':
            print("\nExporting analysis to CSV...")
            analyzer.export_to_csv()
        elif choice == '6':
            print("\nCreating visualizations...")
            analyzer.create_visualizations()
        else:
            print("Invalid choice. Please enter 1-6 or 'quit'.")


if __name__ == "__main__":
    main()