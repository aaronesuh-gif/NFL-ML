import pickle
import numpy as np
from simple_calculator import get_betting_recommendation


def load_model():
    """Load the trained model"""
    try: 
        with open('model/nfl_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(' Model not found! Run training.py first.')
        return None


def get_input():
    """Get all inputs from user"""
    print("\n" + "="*70)
    print("ENTER GAME INFO")
    print("="*70)
    
    # basic info
    home = input('\nHome team: ')
    away = input('Away team: ')
    spread = float(input('Spread (e.g., -3): '))
    over_under = float(input('Over/Under: '))
    week = int(input('Week (1-18): '))
    
    # rolling stats - home team
    print(f'\n{home} last 3 games:')
    h_wins = float(input('  Win rate (0.0-1.0, e.g., 0.67): ') or 0.5)
    h_score = float(input('  Avg points scored: ') or 24)
    h_allowed = float(input('  Avg points allowed: ') or 21)
    
    # rolling stats - away team
    print(f'\n{away} last 3 games:')
    a_wins = float(input('  Win rate (0.0-1.0): ') or 0.5)
    a_score = float(input('  Avg points scored: ') or 24)
    a_allowed = float(input('  Avg points allowed: ') or 21)
    
    # optional
    h_odds = int(input('\nHome odds (default -110): ') or -110)
    a_odds = int(input('Away odds (default -110): ') or -110)
    bankroll = float(input('Bankroll (default 1000): ') or 1000)
    
    return {
        'home': home, 'away': away,
        'spread': spread, 'over_under': over_under, 'week': week,
        'h_wins': h_wins, 'a_wins': a_wins,
        'h_score': h_score, 'a_score': a_score,
        'h_allowed': h_allowed, 'a_allowed': a_allowed,
        'h_odds': h_odds, 'a_odds': a_odds, 'bankroll': bankroll
    }


def predict(model, game):
    """make prediction"""
    # features in order: spread, o/u, week, rolling stats (9 features)
    features = np.array([[
        game['spread'], game['over_under'], game['week'],
        game['h_wins'], game['a_wins'], game['h_wins'] - game['a_wins'],
        game['h_score'], game['a_score'], game['h_score'] - game['a_score'],
        game['h_allowed'], game['a_allowed'], game['a_allowed'] - game['h_allowed']
    ]])
    
    probs = model.predict_proba(features)[0]
    return {'home_prob': probs[1], 'away_prob': probs[0]}


def show_results(game, pred, rec_h, rec_a):
    """display results"""
    print("\n" + "="*70)
    print(f" {game['away']} @ {game['home']}")
    print("="*70)
    
    print(f"\n PREDICTION:")
    print(f"  Home: {pred['home_prob']*100:.1f}%")
    print(f"  Away: {pred['away_prob']*100:.1f}%")
    
    print(f"\n BETTING:")
    print(f"  HOME: EV {rec_h['expected_value']:+.1f}%", end='')
    if rec_h['should_bet']:
        print(f" → yes! BET {rec_h['units']:.1f}u (${rec_h['bet_amount']:.0f})")
    else:
        print(f" → no! {rec_h['reason']}")
    
    print(f"  AWAY: EV {rec_a['expected_value']:+.1f}%", end='')
    if rec_a['should_bet']:
        print(f" → yes! BET {rec_a['units']:.1f}u (${rec_a['bet_amount']:.0f})")
    else:
        print(f" → no! {rec_a['reason']}")
    
    # best bet
    print(f"\n RECOMMENDATION:", end=' ')
    if rec_h['should_bet'] and rec_a['should_bet']:
        best = rec_h if rec_h['expected_value'] > rec_a['expected_value'] else rec_a
        team = game['home'] if best == rec_h else game['away']
        print(f"BET {best['units']:.1f}u on {team.upper()}")
    elif rec_h['should_bet']:
        print(f"BET {rec_h['units']:.1f}u on {game['home'].upper()}")
    elif rec_a['should_bet']:
        print(f"BET {rec_a['units']:.1f}u on {game['away'].upper()}")
    else:
        print("NO BET")
    
    print("="*70 + "\n")


def main():
    """Main program"""
    print("\nNFL BETTING PREDICTION SYSTEM\n")
    
    model = load_model()
    if not model:
        return
    
    while True:
        try:
            game = get_input()
            
            pred = predict(model, game)
            
            rec_h = get_betting_recommendation(pred['home_prob'], game['h_odds'], game['bankroll'])
            rec_a = get_betting_recommendation(pred['away_prob'], game['a_odds'], game['bankroll'])
            
            show_results(game, pred, rec_h, rec_a)
            
            if input("Another? (y/n): ").lower() != 'y':
                break
        except:
            print("Error - try again\n")
    
    print("Good luck! \n")


if __name__ == "__main__":
    main()