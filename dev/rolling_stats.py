# calculating rolling stats for a team, last 3 games points, poitns allowed, and win %

"""
ROLLING STATISTICS - FEATURE ENGINEERING

This calculates how teams have been performing in recent games
"""

import pandas as pd
import numpy as np


def add_rolling_stats(data):
    """
    Add rolling statistics to improve model accuracy
    
    calculates:
    - How many of last 3 games each team won
    - Average points scored in last 3 games
    - Average points allowed in last 3 games
    
    Args:
        data: Your prepared dataframe
    
    Returns:
        data with new rolling stat columns
    """
    
    print("\nAdding rolling statistics...")
    
    # Make a copy
    df = data.copy()
    
    # Sort by date
    df = df.sort_values('schedule_date').reset_index(drop=True)
    
    
    # ========================================================================
    # HOME TEAM ROLLING STATS
    # ========================================================================
    
    print("  Calculating home team stats...")
    
    # group by home team
    home_grouped = df.groupby('team_home')
    
    # rolling wins (last 3 games as home team)
    df['home_wins_last3'] = (
        home_grouped['home_wins']
        .transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
    )
    
    # rolling average points scored (last 3 home games)
    df['home_avg_score_last3'] = (
        home_grouped['score_home']
        .transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
    )
    
    # rolling average points allowed (last 3 home games)
    df['home_avg_allowed_last3'] = (
        home_grouped['score_away']
        .transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
    )
    
    
    # ========================================================================
    # AWAY TEAM ROLLING STATS
    # ========================================================================
    
    print("  Calculating away team stats...")
    
    # group by away team
    away_grouped = df.groupby('team_away')
    
    # create away wins column (inverse of home_wins)
    df['away_wins'] = 1 - df['home_wins']
    
    # rolling wins (last 3 games as away team)
    df['away_wins_last3'] = (
        away_grouped['away_wins']
        .transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
    )
    
    # rolling average points scored (last 3 away games)
    df['away_avg_score_last3'] = (
        away_grouped['score_away']
        .transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
    )
    
    # rolling average points allowed (last 3 away games)
    df['away_avg_allowed_last3'] = (
        away_grouped['score_home']
        .transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
    )
    
    
    # ========================================================================
    # PERFORMANCE DIFFERENTIALS
    # ========================================================================
    
    print("  Calculating differentials...")
    
    # win percentage differential
    df['win_pct_diff'] = df['home_wins_last3'] - df['away_wins_last3']
    
    # offensive differential (who scores more?)
    df['offensive_diff'] = df['home_avg_score_last3'] - df['away_avg_score_last3']
    
    # defensive differential (who allows fewer points?)
    df['defensive_diff'] = df['away_avg_allowed_last3'] - df['home_avg_allowed_last3']
    
    
    print(f"✓ Added rolling statistics!")
    print(f"  (First few games will have NaN for rolling stats - this is normal)")
    
    return df


# ============================================================================
# UPDATED SELECT_FEATURES FUNCTION
# Add this to replace your current select_features()
# ============================================================================

def select_features_with_rolling_stats(data):
    """
    Select features INCLUDING rolling stats
    """
    
    # Original features
    feature_columns = [
        'spread_favorite',
        'over_under_line',
        'schedule_week',
        
        # rolling stats
        'home_wins_last3',
        'away_wins_last3',
        'win_pct_diff',
        
        'home_avg_score_last3',
        'away_avg_score_last3',
        'offensive_diff',
        
        'home_avg_allowed_last3',
        'away_avg_allowed_last3',
        'defensive_diff',
    ]
    
    print("\nSelecting features:")
    for i, col in enumerate(feature_columns):
        print(f"  {i+1}. {col}")
    
    # Convert to numeric
    for col in feature_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Remove rows with missing data
    data_clean = data.dropna(subset=feature_columns + ['home_wins'])
    
    print(f"\n✓ {len(data_clean)} games have all features")
    print(f"  (Removed {len(data) - len(data_clean)} games with missing data)")
    
    # Create X and y
    X = data_clean[feature_columns]
    y = data_clean['home_wins']
    dates = data_clean['schedule_date']
    
    return X, y, dates, data_clean


