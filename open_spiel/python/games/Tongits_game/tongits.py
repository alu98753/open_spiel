# tongits.py
import random
from copy import deepcopy
from itertools import combinations
from collections import defaultdict
import functools

from game import Game, State, Action, Player
from typing import Dict, List, Tuple, Any, Optional, Set

# --- 卡牌與組合輔助函式 ---

SUITS = ['S', 'H', 'D', 'C']
RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
RANK_VALUES_RUN = {r: i + 1 for i, r in enumerate(RANKS)}
RANK_VALUES_SCORE = {r: (i + 1 if i < 9 else 10) for i, r in enumerate(RANKS)}
RANK_VALUES_SCORE['A'] = 1

def _create_deck() -> List[str]:
    """創建一副52張的撲克牌"""
    return [s + r for s in SUITS for r in RANKS]

def _sort_hand(hand: List[str]) -> List[str]:
    """對手牌進行排序，方便檢視和組合判斷"""
    return sorted(hand, key=lambda c: (SUITS.index(c[0]), RANK_VALUES_RUN[c[1]]))

def _is_valid_meld(cards: List[str]) -> bool:
    """檢查一組牌是否為有效的組合 (Set or Run)"""
    if len(cards) < 3:
        return False
    
    cards = _sort_hand(cards)
    
    # 檢查 Set (同數字牌)
    is_set = all(c[1] == cards[0][1] for c in cards)
    if is_set and len(cards) <= 4:
        return True

    # 檢查 Run (順子)
    is_run = all(c[0] == cards[0][0] for c in cards)
    if is_run:
        for i in range(len(cards) - 1):
            if RANK_VALUES_RUN[cards[i+1][1]] - RANK_VALUES_RUN[cards[i][1]] != 1:
                return False
        # A 不能接 K
        if 'K' in [c[1] for c in cards] and 'A' in [c[1] for c in cards]:
             return False # QKA is not a valid run per spec
        return True

    return False

def _is_secret(meld: List[str]) -> bool:
    """檢查一個組合是否為 Secret"""
    if not _is_valid_meld(meld):
        return False
    
    # 四條
    if len(meld) == 4 and all(c[1] == meld[0][1] for c in meld):
        return True
    
    # 長度大於等於 5 的同花順
    is_run = all(c[0] == meld[0][0] for c in meld)
    if is_run and len(meld) >= 5:
        return True
        
    return False

@functools.lru_cache(maxsize=None)
def _find_best_melds_score(hand_tuple: Tuple[str]) -> int:
    """
    遞迴函式，使用快取來尋找手牌中的最佳組合，以計算最低分數。
    """
    hand = list(hand_tuple)
    if not hand:
        return 0

    min_score = sum(RANK_VALUES_SCORE[c[1]] for c in hand)

    # 產生所有可能的組合並遞迴
    for r in range(3, len(hand) + 1):
        for combo in combinations(hand, r):
            if _is_valid_meld(list(combo)):
                remaining_hand = list(hand)
                for card in combo:
                    remaining_hand.remove(card)
                
                score = _find_best_melds_score(tuple(sorted(remaining_hand)))
                if score < min_score:
                    min_score = score
    
    return min_score

def _calculate_hand_score(hand: List[str]) -> int:
    """計算一手牌的最低分數"""
    if not hand:
        return 0
    return _find_best_melds_score(tuple(sorted(hand)))


# --- TongitsGame 類別實作 ---

class TongitsGame(Game):
    NUM_PLAYERS = 3

    @staticmethod
    def init_state() -> State:
        deck = _create_deck()
        random.shuffle(deck)

        players = [f'P{i+1}' for i in range(TongitsGame.NUM_PLAYERS)]
        
        state: State = {
            'game_config': {
                'num_players': TongitsGame.NUM_PLAYERS,
                'players': players,
            },
            'game_state': {
                'cur_player': random.choice(players),
                'ended': False,
                'phase': 'DRAW_PHASE', # DRAW_PHASE, PLAY_PHASE, SHOWDOWN_PHASE
                'end_reason': None, # Tongits, Stock Empty, Showdown
                'winner': None,
                'fighter': None, # The player who called 'fight'
                'showdown_decisions': {}, # For challenge/fold
                'stuck_by': {p: set() for p in players}, # Players who stuck the key player
            },
            'players': {p: {} for p in players},
            'areas': {
                'stock': deck,
                'discard_pile': [],
                **{f'{p}/hand': [] for p in players},
                **{f'{p}/melds': [] for p in players},
            }
        }
        
        # 發牌
        for p in players:
            state['areas'][f'{p}/hand'] = [state['areas']['stock'].pop() for _ in range(13)]
        
        return state

    @staticmethod
    def available_actions(state: State) -> Dict[Player, List[Action]]:
        actions: Dict[Player, List[Action]] = defaultdict(list)
        if state['game_state']['ended']:
            return actions

        phase = state['game_state']['phase']
        cur_player = state['game_state']['cur_player']
        hand = state['areas'][f'{cur_player}/hand']
        
        if phase == 'DRAW_PHASE':
            # Draw
            if state['areas']['stock']:
                actions[cur_player].append(('draw', None))
            
            # Pick
            if state['areas']['discard_pile']:
                top_card = state['areas']['discard_pile'][-1]
                for r in range(2, len(hand) + 1):
                    for combo_from_hand in combinations(hand, r):
                        potential_meld = [top_card] + list(combo_from_hand)
                        if _is_valid_meld(potential_meld):
                            actions[cur_player].append(('pick', {'cards': list(combo_from_hand)}))
            
            # Fight
            can_fight = (
                len(state['areas'][f'{cur_player}/melds']) > 0 and 
                not state['game_state']['stuck_by'][cur_player]
            )
            if can_fight:
                actions[cur_player].append(('fight', None))

        elif phase == 'PLAY_PHASE':
            # Dump
            for card in set(hand): # Use set to avoid duplicate actions
                actions[cur_player].append(('dump', {'card': card}))
            
            # Combine
            for r in range(3, len(hand) + 1):
                for combo in combinations(hand, r):
                    if _is_valid_meld(list(combo)):
                        actions[cur_player].append(('combine', {'cards': list(combo)}))
            
            # Stick
            for p_id in state['game_config']['players']:
                for i, meld in enumerate(state['areas'][f'{p_id}/melds']):
                    for card in hand:
                        if _is_valid_meld(meld + [card]):
                            actions[cur_player].append(('stick', {'card': card, 'target_player': p_id, 'target_meld_index': i}))

        elif phase == 'SHOWDOWN_PHASE':
            fighter = state['game_state']['fighter']
            for p_id in state['game_config']['players']:
                if p_id == fighter: continue
                if p_id in state['game_state']['showdown_decisions']: continue

                # 檢查 Burn 狀態
                is_burned = (
                    len(state['areas'][f'{p_id}/melds']) == 0 and 
                    not any(_is_secret(m) for m in (_find_all_melds_in_hand(state['areas'][f'{p_id}/hand'])))
                )
                if not is_burned:
                    actions[p_id] = [('challenge', None), ('fold', None)]
        
        return dict(actions)

    @staticmethod
    def apply_action(state: State, actions: Dict[Player, Action]) -> State:
        new_state = deepcopy(state)
        
        # Showdown phase can have multiple players acting
        if new_state['game_state']['phase'] == 'SHOWDOWN_PHASE':
            for p_id, (action_name, _) in actions.items():
                new_state['game_state']['showdown_decisions'][p_id] = action_name
            
            # Check if all decisions are made
            fighter = new_state['game_state']['fighter']
            eligible_players = [p for p in new_state['game_config']['players'] if p != fighter]
            # filter out burned players
            eligible_players = [p for p in eligible_players if not (len(new_state['areas'][f'{p}/melds']) == 0 and not any(_is_secret(m) for m in _find_all_melds_in_hand(new_state['areas'][f'{p}/hand'])))]


            if len(new_state['game_state']['showdown_decisions']) == len(eligible_players):
                new_state['game_state']['ended'] = True
                new_state['game_state']['end_reason'] = 'Showdown'
            return new_state


        # Standard turn phases (only one player acts)
        player, (action_name, params) = list(actions.items())[0]
        hand = new_state['areas'][f'{player}/hand']
        
        # --- Action Logic ---
        if action_name == 'draw':
            hand.append(new_state['areas']['stock'].pop())
            new_state['game_state']['phase'] = 'PLAY_PHASE'
        
        elif action_name == 'pick':
            top_card = new_state['areas']['discard_pile'].pop()
            cards_from_hand = params['cards']
            new_meld = [top_card] + cards_from_hand
            for card in cards_from_hand:
                hand.remove(card)
            new_state['areas'][f'{player}/melds'].append(_sort_hand(new_meld))
            new_state['game_state']['phase'] = 'PLAY_PHASE'

        elif action_name == 'fight':
            new_state['game_state']['phase'] = 'SHOWDOWN_PHASE'
            new_state['game_state']['fighter'] = player
            return new_state

        elif action_name == 'combine':
            meld_cards = params['cards']
            for card in meld_cards:
                hand.remove(card)
            new_state['areas'][f'{player}/melds'].append(_sort_hand(list(meld_cards)))
            
        elif action_name == 'stick':
            card = params['card']
            target_player = params['target_player']
            meld_idx = params['target_meld_index']
            hand.remove(card)
            new_state['areas'][f'{target_player}/melds'][meld_idx].append(card)
            new_state['areas'][f'{target_player}/melds'][meld_idx] = _sort_hand(new_state['areas'][f'{target_player}/melds'][meld_idx])
            if player != target_player:
                new_state['game_state']['stuck_by'][target_player].add(player)

        elif action_name == 'dump':
            card = params['card']
            hand.remove(card)
            new_state['areas']['discard_pile'].append(card)
            
            # Check for game end by stock empty
            if not new_state['areas']['stock']:
                new_state['game_state']['ended'] = True
                new_state['game_state']['end_reason'] = 'Stock Empty'
            else:
                # End of turn, move to next player
                players = new_state['game_config']['players']
                current_idx = players.index(player)
                new_state['game_state']['cur_player'] = players[(current_idx + 1) % len(players)]
                new_state['game_state']['phase'] = 'DRAW_PHASE'
                # Reset all 'stuck_by' flags as a full round has passed for this player
                new_state['game_state']['stuck_by'] = {p: set() for p in players}

        # --- Check for Tongits after every move ---
        if _calculate_hand_score(hand) == 0:
            new_state['game_state']['ended'] = True
            new_state['game_state']['end_reason'] = 'Tongits'
            new_state['game_state']['winner'] = player
            
        return new_state

    @staticmethod
    def score(state: State) -> Dict[str, Any]:
        if not state['game_state']['ended']:
            return {}

        players = state['game_config']['players']
        end_reason = state['game_state']['end_reason']
        
        hand_scores = {p: _calculate_hand_score(state['areas'][f'{p}/hand']) for p in players}
        
        # --- Determine Winner ---
        winner = None
        if end_reason == 'Tongits':
            winner = state['game_state']['winner']
        elif end_reason == 'Stock Empty':
            min_score = min(hand_scores.values())
            tied_players = [p for p, s in hand_scores.items() if s == min_score]
            if len(tied_players) == 1:
                winner = tied_players[0]
            else:
                last_player = state['game_state']['cur_player'] # The one who dumped last
                last_player_idx = players.index(last_player)
                for i in range(1, len(players) + 1):
                    check_idx = (last_player_idx + i) % len(players)
                    if players[check_idx] in tied_players:
                        winner = players[check_idx]
                        break
        elif end_reason == 'Showdown':
            fighter = state['game_state']['fighter']
            challengers = [p for p, d in state['game_state']['showdown_decisions'].items() if d == 'challenge']
            contenders = [fighter] + challengers
            
            min_score = min(hand_scores[p] for p in contenders)
            tied_players = [p for p in contenders if hand_scores[p] == min_score]
            if len(tied_players) == 1:
                winner = tied_players[0]
            else:
                fighter_idx = players.index(fighter)
                for i in range(len(players)):
                    check_idx = (fighter_idx - i + len(players)) % len(players)
                    if players[check_idx] in tied_players:
                        winner = players[check_idx]
                        break
        
        # --- Calculate Final Scores ---
        final_scores = {p: 0 for p in players}
        losers = [p for p in players if p != winner]

        # 1. Basic win
        for loser in losers:
            final_scores[loser] -= 1
            final_scores[winner] += 1
            
        # 2. Tongits bonus
        if end_reason == 'Tongits':
            for loser in losers:
                final_scores[loser] -= 1
                final_scores[winner] += 1
        
        # 3. Fight bonus
        if end_reason == 'Showdown':
            challengers = [p for p, d in state['game_state']['showdown_decisions'].items() if d == 'challenge']
            fight_losers = [p for p in ([state['game_state']['fighter']] + challengers) if p != winner]
            for loser in fight_losers:
                final_scores[loser] -= 1
                final_scores[winner] += 1

        # 4. Burn penalty
        burned_players = []
        for p in players:
            has_secret = any(_is_secret(m) for m in _find_all_melds_in_hand(state['areas'][f'{p}/hand']))
            if len(state['areas'][f'{p}/melds']) == 0 and not has_secret:
                burned_players.append(p)
        
        for p_burn in burned_players:
            for p_other in players:
                if p_burn != p_other:
                    final_scores[p_burn] -= 1
                    final_scores[p_other] += 1
                    
        # 5. Secret bonus
        secret_holders = []
        for p in players:
            if any(_is_secret(m) for m in _find_all_melds_in_hand(state['areas'][f'{p}/hand'])):
                 secret_holders.append(p)

        for p_secret in secret_holders:
            for p_other in players:
                if p_secret != p_other:
                    final_scores[p_secret] += 1
                    final_scores[p_other] -= 1

        return {
            'winner': winner,
            'end_reason': end_reason,
            'player_hand_scores': hand_scores,
            'final_scores': final_scores,
            'burned_players': burned_players,
            'secret_holders': secret_holders,
        }

def _find_all_melds_in_hand(hand):
    """Helper to find all possible melds in a hand, used for burn/secret check."""
    melds = []
    for r in range(3, len(hand) + 1):
        for combo in combinations(hand, r):
            if _is_valid_meld(list(combo)):
                melds.append(list(combo))
    return melds


# --- CLI 遊玩介面 ---
def print_state(state):
    cur_player = state['game_state']['cur_player']
    print("\n" + "="*50)
    print(f"Current Turn: {cur_player} | Phase: {state['game_state']['phase']}")
    print(f"Stock: {len(state['areas']['stock'])} cards | Discard Top: {state['areas']['discard_pile'][-1] if state['areas']['discard_pile'] else 'Empty'}")
    print("-"*50)
    
    for p in state['game_config']['players']:
        hand = _sort_hand(state['areas'][f'{p}/hand'])
        melds = state['areas'][f'{p}/melds']
        print(f"Player {p}:")
        if p == cur_player or state['game_state']['ended']:
            print(f"  Hand ({len(hand)} cards, score={_calculate_hand_score(hand)}): {' '.join(hand)}")
        else:
             print(f"  Hand ({len(hand)} cards)")
        print(f"  Melds: {melds if melds else 'None'}")
    print("="*50)

def cli_play():
    state = TongitsGame.init_state()
    
    while not state['game_state']['ended']:
        print_state(state)
        
        available = TongitsGame.available_actions(state)
        
        if not any(available.values()):
            print("No actions available. This might be a bug or an unresolved game state.")
            break
            
        actions_to_apply = {}

        if state['game_state']['phase'] == 'SHOWDOWN_PHASE':
            print("--- SHOWDOWN ---")
            fighter = state['game_state']['fighter']
            print(f"{fighter} has called a FIGHT!")
            for p_id, p_actions in available.items():
                print(f"Player {p_id}, what is your decision?")
                for i, (name, _) in enumerate(p_actions):
                    print(f"  {i+1}: {name}")
                
                choice = -1
                while choice < 1 or choice > len(p_actions):
                    try:
                        choice = int(input(f"Enter choice for {p_id} (1-{len(p_actions)}): "))
                    except ValueError:
                        choice = -1
                actions_to_apply[p_id] = p_actions[choice-1]

        else: # Normal turn
            cur_player = state['game_state']['cur_player']
            player_actions = available[cur_player]
            
            print(f"Available actions for {cur_player}:")
            for i, (name, params) in enumerate(player_actions):
                if name in ['pick', 'combine', 'dump', 'stick']:
                    print(f"  {i+1}: {name} {params}")
                else:
                    print(f"  {i+1}: {name}")

            choice = -1
            while choice < 1 or choice > len(player_actions):
                try:
                    choice = int(input(f"Enter your choice (1-{len(player_actions)}): "))
                except ValueError:
                    choice = -1
            
            actions_to_apply[cur_player] = player_actions[choice-1]

        state = TongitsGame.apply_action(state, actions_to_apply)

    # Game Over
    print("\n" + "#"*20 + " GAME OVER " + "#"*20)
    print_state(state)
    
    score_info = TongitsGame.score(state)
    print("\n--- FINAL RESULTS ---")
    print(f"Reason for game end: {score_info['end_reason']}")
    print(f"Winner: {score_info['winner']}")
    print("\nHand Scores:")
    for p, s in score_info['player_hand_scores'].items():
        print(f"  {p}: {s}")
        
    print(f"\nBurned Players: {score_info['burned_players'] if score_info['burned_players'] else 'None'}")
    print(f"Players with Secret in hand: {score_info['secret_holders'] if score_info['secret_holders'] else 'None'}")

    print("\nFinal Score changes:")
    for p, s in score_info['final_scores'].items():
        print(f"  {p}: {s:+} points")


if __name__ == '__main__':
    cli_play()