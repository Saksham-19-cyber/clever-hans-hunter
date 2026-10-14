"""
tests/test_game_loop.py  –  Unit tests for streamlit_app.py game-loop functions.

These tests import only the pure-logic helpers (game_train, game_leaderboard_append,
game_is_unlocked) — they do NOT start the Streamlit server.

Run with:
    python -m pytest tests/ -v
"""

import sys 
import os 
sys .path .insert (0 ,os .path .join (os .path .dirname (__file__ ),".."))


import types 
import unittest .mock as mock 


_st_stub =types .ModuleType ("streamlit")
_st_stub .cache_resource =lambda **kw :(lambda fn :fn )
_st_stub .session_state ={}
sys .modules .setdefault ("streamlit",_st_stub )

from streamlit_app import (
game_train ,game_leaderboard_append ,game_is_unlocked ,
LEVEL_TARGETS ,cfg_summary ,
)
from train import TrainConfig ,TrainResult 
from data import build_datasets 
import numpy as np 


def _quick_cfg (level :int )->TrainConfig :
    """Minimal 1-epoch head-only config for fast tests."""
    return TrainConfig (
    finetune_mode ="head_only",
    optimizer ="adam",
    learning_rate =1e-3 ,
    epochs =1 ,
    loss_fn_name ="cross_entropy",
    seed =42 ,
    )


def _fake_result (wg_acc :float )->TrainResult :
    """Construct a minimal TrainResult with a given worst-group accuracy."""
    return TrainResult (
    val_acc =0.9 ,
    test_avg_acc =0.75 ,
    worst_group_acc =wg_acc ,
    group_accs ={0 :wg_acc ,1 :0.8 ,2 :0.7 ,3 :0.85 },
    confusion_matrix =np .array ([[40 ,10 ],[5 ,45 ]]),
    history =[{"epoch":1 ,"train_acc":0.7 ,"val_acc":0.9 ,
    "train_loss":0.3 }],
    model =None ,
    suspicious =[],
    )


def test_game_train_returns_worst_group ():
    """game_train(level=1, cfg) must return a TrainResult with a valid worst-group acc."""
    cfg =_quick_cfg (1 )
    result =game_train (1 ,cfg )
    assert isinstance (result .worst_group_acc ,float ),(
    f"worst_group_acc type: {type (result .worst_group_acc )}"
    )
    assert 0.0 <=result .worst_group_acc <=1.0 ,(
    f"worst_group_acc out of range: {result .worst_group_acc }"
    )


def test_game_train_val_acc_in_range ():
    """Val accuracy must be in [0, 1]."""
    cfg =_quick_cfg (1 )
    result =game_train (1 ,cfg )
    assert 0.0 <=result .val_acc <=1.0 


def test_game_train_history_has_entries ():
    """History must contain one entry per epoch."""
    cfg =_quick_cfg (1 )
    result =game_train (1 ,cfg )
    assert len (result .history )==cfg .epochs 


def test_leaderboard_append_increases_length ():
    board =[]
    cfg =_quick_cfg (1 )
    r1 =_fake_result (0.60 )
    board =game_leaderboard_append (board ,1 ,r1 ,cfg )
    assert len (board )==1 


def test_leaderboard_sorted_descending ():
    board =[]
    cfg =_quick_cfg (1 )
    for wg in [0.50 ,0.75 ,0.62 ]:
        r =_fake_result (wg )
        board =game_leaderboard_append (board ,1 ,r ,cfg )
    scores =[e [1 ]for e in board ]
    assert scores ==sorted (scores ,reverse =True ),(
    f"Leaderboard not sorted: {scores }"
    )


def test_leaderboard_stores_level ():
    board =[]
    cfg1 =_quick_cfg (1 )
    cfg2 =_quick_cfg (2 )
    r1 =_fake_result (0.60 )
    r2 =_fake_result (0.55 )
    board =game_leaderboard_append (board ,1 ,r1 ,cfg1 )
    board =game_leaderboard_append (board ,2 ,r2 ,cfg2 )
    levels_stored ={e [0 ]for e in board }
    assert levels_stored =={1 ,2 }


def test_unlock_at_target ():
    for level in [1 ,2 ,3 ]:
        target =LEVEL_TARGETS [level ]
        r_pass =_fake_result (target )
        r_fail =_fake_result (target -0.01 )
        assert game_is_unlocked (r_pass ,level ),(
        f"Level {level }: should unlock at {target :.0%}"
        )
        assert not game_is_unlocked (r_fail ,level ),(
        f"Level {level }: should NOT unlock at {target -0.01 :.0%}"
        )


def test_unlock_boundary_exact ():
    """Exactly at the target should unlock."""
    for level in [1 ,2 ,3 ]:
        target =LEVEL_TARGETS [level ]
        r_exact =_fake_result (target )
        assert game_is_unlocked (r_exact ,level )


def test_cfg_summary_contains_aug_names ():
    cfg =TrainConfig (
    use_color_jitter =True ,
    use_random_erasing =True ,
    use_hflip =False ,
    loss_fn_name ="group_reweighted_ce",
    finetune_mode ="last_block",
    epochs =2 ,
    learning_rate =5e-4 ,
    )
    s =cfg_summary (cfg )
    assert "CJ"in s 
    assert "Erase"in s 
    assert "Flip"not in s 
    assert "group_reweighted_ce"in s 
    assert "last_block"in s 


if __name__ =="__main__":
    import traceback 
    tests =[
    test_game_train_returns_worst_group ,
    test_game_train_val_acc_in_range ,
    test_game_train_history_has_entries ,
    test_leaderboard_append_increases_length ,
    test_leaderboard_sorted_descending ,
    test_leaderboard_stores_level ,
    test_unlock_at_target ,
    test_unlock_boundary_exact ,
    test_cfg_summary_contains_aug_names ,
    ]
    passed =failed =0 
    for fn in tests :
        try :
            fn ()
            print (f"  PASS  {fn .__name__ }")
            passed +=1 
        except Exception as e :
            print (f"  FAIL  {fn .__name__ }  -->  {e }")
            traceback .print_exc ()
            failed +=1 
    print (f"\n{passed } passed / {failed } failed")
    sys .exit (0 if failed ==0 else 1 )
