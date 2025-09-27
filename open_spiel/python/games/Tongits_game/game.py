from abc import ABC
from typing import Dict, List, Tuple, Any, Optional


State = Dict[str, Any]
Action = Tuple[str, Optional[Dict[str, Any]]]
Player = str

class Game(ABC):
    @staticmethod
    def init_state() -> State:
        '''
        建議格式為：
        state: Dict[str, Any] = {
            'game_config':{
                'num_players': num_players,
                其他隨遊戲設計所需的規則參數
            },
            'game_state':{
                'cur_player': cur_player,
                'ended': False,
                'phase': phase,
                'end_reason': None,
                記錄遊戲流程相關的資訊，也是隨遊戲設計去做調整
            }
            'players':{
                'player_id': { 玩家的狀態，依照玩家類型不同可以不一樣，但實體的手牌持有之類的不放這裡}
                玩家 id 的格式依照遊戲不同以方便設計的主要概念
                next_player 之類的東西也可以放這裡面
                可以把 player 的私有 area 也指過來，方便使用為主
            },
            'areas':{ 主要用來儲存實際上交易的物件，若有透過推論或記錄的資訊，建議放在 players 裡面
                'area_name': List[Any] 隨遊戲設計需求可以修改區域的內容
                'player_id/area_name': List[Any] 手牌之類的，有持有者的 area
                'area_name/palyer_id': ，也可以反過來變成這個形式
            },
        }
        '''
        raise NotImplementedError()

    @staticmethod
    def available_actions(state: State) -> Dict[Player, List[Action]]:
        '''
        回傳可行動作：
        {
            player_id: [('action_name', **action_params)],
        }
        '''
        raise NotImplementedError()

    @staticmethod
    def apply_action(state: State, actions: Dict[Player, Action]) -> State:
        """
        不更動既有 state 的狀態下，產生 action 後的新 state
        """
        raise NotImplementedError()

    @staticmethod
    def score(state: State) -> Dict[str, Any]:
        """
        讀取遊戲狀態，計算勝負與計分，依遊戲設計回傳所需的計分結果
        """
        raise NotImplementedError()