

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from simple_calculator import get_betting_recommendation


# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

def load_model():
    """Load trained model"""
    with open('model/nfl_model.pkl', 'rb') as f:
        return pickle.load(f)


def load_test_data():
    """
    Load the most recent 2 seasons only
    """
    # Load raw data
    df = pd.read_csv('data/spreadspoke_scores.csv')
    
    # Prepare it (same as training.py)
    df['home_wins'] = (df['score_home'] > df['score_away']).astype(int)
    df['schedule_date'] = pd.to_datetime(df['schedule_date'])
    df = df.dropna(subset=['score_home', 'score_away'])
    
    # Get the 2 most recent complete seasons
    # Extract year from schedule_date
    df['year'] = df['schedule_date'].dt.year
    
    # Get unique years and take the 2 most recent
    all_years = sorted(df['year'].unique())
    recent_2_years = all_years[-2:]  # Last 2 years
    
    print(f"  Using seasons: {recent_2_years}")
    
    # Filter to only those 2 years
    df = df[df['year'].isin(recent_2_years)]
    
    print(f"  Total games in these seasons: {len(df)}")
    
    # Add rolling stats
    from rolling_stats import add_rolling_stats, select_features_with_rolling_stats
    df = add_rolling_stats(df)
    
    # Select features
    X, y, dates, clean_data = select_features_with_rolling_stats(df)
    
    print(f"  Games with complete features: {len(X)}")
    
    return X, y, dates, clean_data


# ============================================================================
# BACKTEST THE MODEL
# ============================================================================

def backtest_model(model, X_test, y_test, dates_test, clean_test, starting_bankroll=1000):
    """
    Simulate betting on every game in test set
    Track profit over time
    """
    
    print("Running backtest on test games...")
    
    results = []
    bankroll = starting_bankroll
    
    for idx in range(len(X_test)):
        # Get game data
        features = X_test.iloc[idx:idx+1].values
        actual_winner = y_test.iloc[idx]  # 1 = home won, 0 = away won
        date = dates_test.iloc[idx]
        game_data = clean_test.iloc[idx]
        
        # Get model prediction
        probs = model.predict_proba(features)[0]
        home_prob = probs[1]
        away_prob = probs[0]
        
        # Assume -110 odds for both sides (typical)
        home_odds = -110
        away_odds = -110
        
        # Get betting recommendations
        home_rec = get_betting_recommendation(home_prob, home_odds, bankroll)
        away_rec = get_betting_recommendation(away_prob, away_odds, bankroll)
        
        # Determine bet (pick best EV if any)
        bet_made = None
        bet_amount = 0
        bet_on_home = None
        
        if home_rec['should_bet'] and away_rec['should_bet']:
            # Both positive EV - pick better one
            if home_rec['expected_value'] > away_rec['expected_value']:
                bet_made = True
                bet_on_home = True
                bet_amount = home_rec['bet_amount']
            else:
                bet_made = True
                bet_on_home = False
                bet_amount = away_rec['bet_amount']
        elif home_rec['should_bet']:
            bet_made = True
            bet_on_home = True
            bet_amount = home_rec['bet_amount']
        elif away_rec['should_bet']:
            bet_made = True
            bet_on_home = False
            bet_amount = away_rec['bet_amount']
        
        # Calculate profit/loss
        profit = 0
        won_bet = False
        
        if bet_made:
            if bet_on_home and actual_winner == 1:
                won_bet = True
                profit = bet_amount * (100 / 110)  # Win $100 for every $110 bet
                bankroll += profit
            elif not bet_on_home and actual_winner == 0:
                won_bet = True
                profit = bet_amount * (100 / 110)
                bankroll += profit
            else:
                won_bet = False
                profit = -bet_amount  # Lose your bet
                bankroll += profit  # This subtracts from bankroll
                # Make sure bankroll doesn't go negative
                if bankroll < 0:
                    bankroll = 0
        
        # Record result
        results.append({
            'date': date,
            'home_team': game_data['team_home'],
            'away_team': game_data['team_away'],
            'home_prob': home_prob,
            'away_prob': away_prob,
            'actual_winner': 'home' if actual_winner == 1 else 'away',
            'bet_made': bet_made,
            'bet_on': 'home' if bet_on_home else 'away' if bet_made else None,
            'bet_amount': bet_amount,
            'won_bet': won_bet if bet_made else None,
            'profit': profit,
            'bankroll': bankroll
        })
    
    df_results = pd.DataFrame(results)
    
    print(f"\n✓ Backtest complete!")
    print(f"  Games analyzed: {len(results)}")
    print(f"  Bets placed: {df_results['bet_made'].sum()}")
    print(f"  Bets won: {df_results['won_bet'].sum()}")
    print(f"  Win rate: {df_results['won_bet'].sum() / df_results['bet_made'].sum() * 100:.1f}%")
    print(f"  Starting bankroll: ${starting_bankroll}")
    print(f"  Ending bankroll: ${bankroll:.2f}")
    print(f"  Total profit: ${bankroll - starting_bankroll:.2f}")
    print(f"  ROI: {(bankroll - starting_bankroll) / starting_bankroll * 100:.1f}%")
    
    return df_results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_bankroll_over_time(df_results, starting_bankroll=1000):
    """Plot how bankroll changes over time"""
    
    plt.figure(figsize=(12, 6))
    
    # Plot bankroll
    plt.plot(df_results['date'], df_results['bankroll'], linewidth=2, color='#2E86AB')
    
    # Add starting line
    plt.axhline(y=starting_bankroll, color='red', linestyle='--', 
                label=f'Starting: ${starting_bankroll}', alpha=0.7)
    
    # Styling
    plt.title('Bankroll Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Bankroll ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format
    plt.tight_layout()
    
    return plt





# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run backtest and generate graphs"""
    
    print("="*70)
    print("NFL MODEL BACKTESTING & VISUALIZATION")
    print("="*70)
    
    # Load model and data
    print("\nLoading model and test data...")
    model = load_model()
    X_test, y_test, dates_test, clean_test = load_test_data()
    
    # Run backtest
    starting_bankroll = 1000
    results = backtest_model(model, X_test, y_test, dates_test, clean_test, starting_bankroll)
    
    # Generate graphs
    plot_all_graphs(results, starting_bankroll)
    
    # Show graphs
    print("\nOpening graphs...")
    plt.show()
    
    print("\n" + "="*70)
    print("BACKTEST COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()