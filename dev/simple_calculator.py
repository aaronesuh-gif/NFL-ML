#simple betting calc

def american_to_decimal(american_odds):
    'convert oddsto decimals, -110 --> 1.909 and +150 ----> 2.50'

    if american_odds > 0:
        return (american_odds/100) + 1
    else:
        return (100/abs(american_odds)) + 1
    



def calculate_vegas_probability(american_odds):

    'vegas think prob, -110 ---> 52.4%, +150 ---> 40.0%'
    if american_odds > 0:
        return 100/(american_odds+100)
    else:
        return abs(american_odds)/ (abs(american_odds)+ 100)
    

def calculate_ev(your_probability,vegas_odds):
    'calculating ev, tells if bet is profitable long term'
    'positive = good, negative = bad'

    #convert odds to decimal number/odds
    decimal_odds = american_to_decimal(vegas_odds)

    #how much profit per $ if you win

    profit_if_win = decimal_odds - 1

    ev = (your_probability + profit_if_win) - ((1 - your_probability) * 1)

    #ev percentage
    return ev * 100

def calculate_units_to_bet(your_probability,vegas_odds):
    # 1 unit = 1% of bankroll, sizing by kelly criterion

    decimal_odds = american_to_decimal(vegas_odds)

    b = decimal_odds - 1
    p = your_probability
    q = 1- p

    kelly_percentage = (b * p -  q)/b

    kelly_percentage = kelly_percentage * 0.25 #quartering to be safer

    units = kelly_percentage * 100


    #dont bet if negative or more than 5 units
    if units <= 0:
        return 0
    elif units > 5:
        return 5
    else: 
        return units
    

def get_betting_recommendation(your_probability,vegas_odds,bankroll= 1000):

    vegas_prob = calculate_vegas_probability(vegas_odds)
    edge = your_probability - vegas_prob
    ev = calculate_ev(your_probability,vegas_odds)
    units = calculate_units_to_bet(your_probability,vegas_odds)
    bet_amount = units * (bankroll/100)

    #shoudl we bet

    should_bet = False

    if ev < 1.0:
        reason = 'ev too low'
    elif your_probability < 0.51:
        reason = 'confidence too low'
    elif units == 0:
        reason = 'no edge'
    else:
        should_bet = True
        reason = 'good bet'

    return {
        'your_probability': your_probability * 100,
        'vegas_probability': vegas_prob * 100,
        'edge': edge * 100,
        'expected_value': ev,
        'units': units,
        'bet_amount': bet_amount,
        'should_bet': should_bet,
        'reason': reason
    }

def print_recommendation(rec):
    'print recs'

    print(f"\n  Your Model: {rec['your_probability']:.1f}%")
    print(f"  Vegas:      {rec['vegas_probability']:.1f}%")
    print(f"  Your Edge:  {rec['edge']:+.1f}%")
    print(f"  EV:         {rec['expected_value']:+.2f}%")
    
    if rec['should_bet']:
        print(f"\n BET {rec['units']:.1f} units (${rec['bet_amount']:.2f})")
    else:
        print(f"\n NO BET - {rec['reason']}")
