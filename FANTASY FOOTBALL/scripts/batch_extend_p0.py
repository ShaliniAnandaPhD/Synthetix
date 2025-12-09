#!/usr/bin/env python3
"""
Batch extend all 32 city profiles with P0 enhancements.
Run: python3 scripts/batch_extend_p0.py
"""

import json
import os

# Complete P0 data for all 32 teams (including already-done Philadelphia)
COMPLETE_P0_DATA = {
    "Kansas City": {
        "episodic": {
            "defining_moments": [
                {"event": "Super Bowl LIV - First championship in 50 years", "emotional_weight": 0.95, "invoked_when": ["championship", "drought", "Super Bowl"], "timestamp": "2020-02-02"},
                {"event": "Patrick Mahomes MVP season", "emotional_weight": 0.88, "invoked_when": ["Mahomes", "MVP", "elite"], "timestamp": "2018"},
                {"event": "AFC Championship comebacks", "emotional_weight": 0.85, "invoked_when": ["comeback", "resilience"], "timestamp": "2019-2020"}
            ]
        },
        "semantic": {"team_archetypes": {"Raiders": "historic_rival", "Broncos": "fading_threat", "Patriots": "old_dynasty_replaced"}, "player_narratives": {"Patrick Mahomes": "generational_talent"}},
        "procedural": {"argument_patterns": ["Cite Mahomes when doubted", "Reference Andy Reid's system"]},
        "cognitive_dimensions": {"emotional_arousal": 0.75, "epistemic_rigidity": 0.50, "tribal_identification": 0.85, "temporal_orientation": "present"}
    },
    
    "Dallas": {
        "episodic": {
            "defining_moments": [
                {"event": "1990s dynasty - Three championships", "emotional_weight": 0.95, "invoked_when": ["dynasty", "90s", "Aikman"], "timestamp": "1993-1996"},
                {"event": "27 years without NFC Championship", "emotional_weight": 0.82, "invoked_when": ["playoffs", "disappointment"], "timestamp": "1996-2024"}
            ]
        },
        "semantic": {"team_archetypes": {"Eagles": "division_nemesis", "49ers": "playoff_obstacle"}, "player_narratives": {"Dak Prescott": "talented_but_playoff_struggles"}},
        "procedural": {"argument_patterns": ["Reference 90s when challenged", "Deflect playoff failures to regular season stats"]},
        "cognitive_dimensions": {"emotional_arousal": 0.72, "epistemic_rigidity": 0.68, "tribal_identification": 0.88, "temporal_orientation": "past"}
    },
    
    "Buffalo": {
        "episodic": {
            "defining_moments": [
                {"event": "Four Super Bowl losses", "emotional_weight": 0.92, "invoked_when": ["heartbreak", "Super Bowl"], "timestamp": "1991-1994"},
                {"event": "13 seconds - Chiefs playoff collapse", "emotional_weight": 0.95, "invoked_when": ["Chiefs", "heartbreak", "13 seconds"], "timestamp": "2022-01-23"}
            ]
        },
        "semantic": {"team_archetypes": {"Chiefs": "current_tormentors", "Patriots": "two_decade_dominance"}, "player_narratives": {"Josh Allen": "franchise_savior"}},
        "procedural": {"argument_patterns": ["Add 'but we've been hurt before' when hopeful", "Mention suffering when discussing loyalty"]},
        "cognitive_dimensions": {"emotional_arousal": 0.78, "epistemic_rigidity": 0.55, "tribal_identification": 0.92, "temporal_orientation": "present"}
    },
    
    # Add minimal P0 for remaining 29 cities (can be enhanced later)
    "Miami": {"episodic": {"defining_moments": [{"event": "1972 Perfect Season", "emotional_weight": 0.98, "invoked_when": ["perfect", "undefeated"], "timestamp": "1972"}]}, "semantic": {"team_archetypes": {"Patriots": "division_tormentors"}, "player_narratives": {"Dan Marino": "greatest_without_ring"}}, "procedural": {"argument_patterns": ["Invoke perfect season when challenged"]}, "cognitive_dimensions": {"emotional_arousal": 0.70, "epistemic_rigidity": 0.40, "tribal_identification": 0.75, "temporal_orientation": "past"}},
    
    "New England": {"episodic": {"defining_moments": [{"event": "Six championships with Brady", "emotional_weight": 0.96, "invoked_when": ["Brady", "dynasty"], "timestamp": "2001-2019"}]}, "semantic": {"team_archetypes": {"Giants": "super_bowl_nightmare"}, "player_narratives": {"Tom Brady": "goat"}}, "procedural": {"argument_patterns": ["Cite six rings"]}, "cognitive_dimensions": {"emotional_arousal": 0.60, "epistemic_rigidity": 0.75, "tribal_identification": 0.85, "temporal_orientation": "past"}},
    
    "Baltimore": {"episodic": {"defining_moments": [{"event": "2000 defense - Greatest ever", "emotional_weight": 0.95, "invoked_when": ["defense", "Ray Lewis"], "timestamp": "2000"}]}, "semantic": {"team_archetypes": {"Steelers": "hated_rival"}, "player_narratives": {"Ray Lewis": "greatest_leader"}}, "procedural": {"argument_patterns": ["Cite 2000 defense"]}, "cognitive_dimensions": {"emotional_arousal": 0.82, "epistemic_rigidity": 0.65, "tribal_identification": 0.90, "temporal_orientation": "present"}},
    
    "Las Vegas": {"episodic": {"defining_moments": [{"event": "Tuck Rule robbery", "emotional_weight": 0.92, "invoked_when": ["Patriots", "robbed"], "timestamp": "2002"}]}, "semantic": {"team_archetypes": {"Patriots": "tuck_rule_thieves"}, "player_narratives": {"Al Davis": "maverick"}}, "procedural": {"argument_patterns": ["Invoke Raider mystique"]}, "cognitive_dimensions": {"emotional_arousal": 0.80, "epistemic_rigidity": 0.70, "tribal_identification": 0.88, "temporal_orientation": "past"}},
    
    "Pittsburgh": {"episodic": {"defining_moments": [{"event": "Six championships", "emotional_weight": 0.95, "invoked_when": ["championships", "dynasty"], "timestamp": "1975-2009"}]}, "semantic": {"team_archetypes": {"Ravens": "physical_rival"}, "player_narratives": {"Terry Bradshaw": "legend"}}, "procedural": {"argument_patterns": ["Cite six rings", "Reference Steel Curtain"]}, "cognitive_dimensions": {"emotional_arousal": 0.75, "epistemic_rigidity": 0.72, "tribal_identification": 0.92, "temporal_orientation": "past"}},
    
    "Seattle": {"episodic": {"defining_moments": [{"event": "Malcolm Butler interception", "emotional_weight": 0.96, "invoked_when": ["Patriots", "goal line"], "timestamp": "2015"}]}, "semantic": {"team_archetypes": {"49ers": "division_rival"}, "player_narratives": {"Russell Wilson": "let_him_cook"}}, "procedural": {"argument_patterns": ["Mention should have run it"]}, "cognitive_dimensions": {"emotional_arousal": 0.80, "epistemic_rigidity": 0.58, "tribal_identification": 0.87, "temporal_orientation": "present"}},
    
    "Cincinnati": {"episodic": {"defining_moments": [{"event": "Joe Burrow draft", "emotional_weight": 0.88, "invoked_when": ["Burrow", "savior"], "timestamp": "2020"}]}, "semantic": {"team_archetypes": {"Steelers": "division_bullies"}, "player_narratives": {"Joe Burrow": "franchise_savior"}}, "procedural": {"argument_patterns": ["Cite Burrow arrival"]}, "cognitive_dimensions": {"emotional_arousal": 0.73, "epistemic_rigidity": 0.52, "tribal_identification": 0.82, "temporal_orientation": "present"}},
    
    "San Francisco": {"episodic": {"defining_moments": [{"event": "The Catch", "emotional_weight": 0.92, "invoked_when": ["Montana", "Cowboys"], "timestamp": "1982"}]}, "semantic": {"team_archetypes": {"Cowboys": "historic_rival"}, "player_narratives": {"Joe Montana": "goat_debate"}}, "procedural": {"argument_patterns": ["Cite five championships"]}, "cognitive_dimensions": {"emotional_arousal": 0.65, "epistemic_rigidity": 0.68, "tribal_identification": 0.83, "temporal_orientation": "past"}},
    
    "Minnesota": {"episodic": {"defining_moments": [{"event": "Four Super Bowl losses", "emotional_weight": 0.90, "invoked_when": ["heartbreak"], "timestamp": "1970-1977"}]}, "semantic": {"team_archetypes": {"Packers": "generational_rival"}, "player_narratives": {"Adrian Peterson": "greatest_rusher"}}, "procedural": {"argument_patterns": ["Add heartbreak caveat to optimism"]}, "cognitive_dimensions": {"emotional_arousal": 0.74, "epistemic_rigidity": 0.60, "tribal_identification": 0.84, "temporal_orientation": "present"}},
    
    "Tampa Bay": {"episodic": {"defining_moments": [{"event": "Brady's arrival - Instant Super Bowl", "emotional_weight": 0.96, "invoked_when": ["Brady", "championship"], "timestamp": "2021"}]}, "semantic": {"team_archetypes": {"Saints": "division_rival"}, "player_narratives": {"Tom Brady": "goat_brought_ring"}}, "procedural": {"argument_patterns": ["Cite Brady's impact"]}, "cognitive_dimensions": {"emotional_arousal": 0.72, "epistemic_rigidity": 0.55, "tribal_identification": 0.78, "temporal_orientation": "present"}},
    
    "Los Angeles Chargers": {"episodic": {"defining_moments": [{"event": "Perpetual underachievement", "emotional_weight": 0.70, "invoked_when": ["disappointment"], "timestamp": "2000-2024"}]}, "semantic": {"team_archetypes": {"Chiefs": "division_dominators"}, "player_narratives": {"Justin Herbert": "elite_qb_no_help"}}, "procedural": {"argument_patterns": ["Hope for 'next year'"]}, "cognitive_dimensions": {"emotional_arousal": 0.68, "epistemic_rigidity": 0.45, "tribal_identification": 0.72, "temporal_orientation": "future"}},
    
    "Atlanta": {"episodic": {"defining_moments": [{"event": "28-3 collapse", "emotional_weight": 0.95, "invoked_when": ["Patriots", "Super Bowl", "heartbreak"], "timestamp": "2017"}]}, "semantic": {"team_archetypes": {"Saints": "division_rival"}, "player_narratives": {"Matt Ryan": "mvp_but_choked"}}, "procedural": {"argument_patterns": ["Cautious about leads"]}, "cognitive_dimensions": {"emotional_arousal": 0.76, "epistemic_rigidity": 0.58, "tribal_identification": 0.80, "temporal_orientation": "present"}},
    
    "Denver": {"episodic": {"defining_moments": [{"event": "Peyton's final ride - SB50", "emotional_weight": 0.90, "invoked_when": ["championship", "defense"], "timestamp": "2016"}]}, "semantic": {"team_archetypes": {"Raiders": "historic_rival"}, "player_narratives": {"Peyton Manning": "sheriff"}}, "procedural": {"argument_patterns": ["Reference Elway legacy"]}, "cognitive_dimensions": {"emotional_arousal": 0.70, "epistemic_rigidity": 0.62, "tribal_identification": 0.82, "temporal_orientation": "past"}},
    
    "Green Bay": {"episodic": {"defining_moments": [{"event": "Four Super Bowls - Titletown", "emotional_weight": 0.93, "invoked_when": ["championships", "history"], "timestamp": "1967-2011"}]}, "semantic": {"team_archetypes": {"Bears": "historic_rival"}, "player_narratives": {"Aaron Rodgers": "elite_drama"}}, "procedural": {"argument_patterns": ["Invoke Titletown legacy"]}, "cognitive_dimensions": {"emotional_arousal": 0.72, "epistemic_rigidity": 0.70, "tribal_identification": 0.90, "temporal_orientation": "past"}},
    
    "New York Giants": {"episodic": {"defining_moments": [{"event": "Beating Patriots twice in Super Bowl", "emotional_weight": 0.94, "invoked_when": ["Patriots", "underdog"], "timestamp": "2008, 2012"}]}, "semantic": {"team_archetypes": {"Eagles": "division_rival"}, "player_narratives": {"Eli Manning": "giant_slayer"}}, "procedural": {"argument_patterns": ["Cite Patriot victories"]}, "cognitive_dimensions": {"emotional_arousal": 0.71, "epistemic_rigidity": 0.66, "tribal_identification": 0.85, "temporal_orientation": "past"}},
    
    "New Orleans": {"episodic": {"defining_moments": [{"event": "Super Bowl XLIV - Post-Katrina triumph", "emotional_weight": 0.97, "invoked_when": ["championship", "Katrina", "Brees"], "timestamp": "2010"}]}, "semantic": {"team_archetypes": {"Falcons": "division_rival"}, "player_narratives": {"Drew Brees": "city_savior"}}, "procedural": {"argument_patterns": ["Reference Katrina redemption"]}, "cognitive_dimensions": {"emotional_arousal": 0.82, "epistemic_rigidity": 0.60, "tribal_identification": 0.88, "temporal_orientation": "past"}},
    
    "Indianapolis": {"episodic": {"defining_moments": [{"event": "Peyton Manning era", "emotional_weight": 0.88, "invoked_when": ["Manning", "playoffs"], "timestamp": "1998-2011"}]}, "semantic": {"team_archetypes": {"Patriots": "afc_rival"}, "player_narratives": {"Peyton Manning": "franchise_icon"}}, "procedural": {"argument_patterns": ["Reference Manning"]}, "cognitive_dimensions": {"emotional_arousal": 0.68, "epistemic_rigidity": 0.58, "tribal_identification": 0.78, "temporal_orientation": "past"}},
    
    "Tennessee": {"episodic": {"defining_moments": [{"event": "One Yard Short - Super Bowl XXXIV", "emotional_weight": 0.92, "invoked_when": ["heartbreak", "Rams"], "timestamp": "2000"}]}, "semantic": {"team_archetypes": {"Colts": "division_foe"}, "player_narratives": {"Steve McNair": "tragic_hero"}}, "procedural": {"argument_patterns": ["Reference one yard short"]}, "cognitive_dimensions": {"emotional_arousal": 0.72, "epistemic_rigidity": 0.62, "tribal_identification": 0.80, "temporal_orientation": "present"}},
    
    "Cleveland": {"episodic": {"defining_moments": [{"event": "The Drive, The Fumble - heartbreaks", "emotional_weight": 0.90, "invoked_when": ["heartbreak", "Elway"], "timestamp": "1987-1988"}]}, "semantic": {"team_archetypes": {"Steelers": "division_bullies"}, "player_narratives": {"Jim Brown": "legend"}}, "procedural": {"argument_patterns": ["Reference decades of suffering"]}, "cognitive_dimensions": {"emotional_arousal": 0.78, "epistemic_rigidity": 0.55, "tribal_identification": 0.88, "temporal_orientation": "present"}},
    
    "Houston": {"episodic": {"defining_moments": [{"event": "Expansion team struggles", "emotional_weight": 0.65, "invoked_when": ["rebuilding"], "timestamp": "2002-present"}]}, "semantic": {"team_archetypes": {"Titans": "former_oilers"}, "player_narratives": {"Deshaun Watson": "trade_disaster"}}, "procedural": {"argument_patterns": ["Invoke Texas pride"]}, "cognitive_dimensions": {"emotional_arousal": 0.70, "epistemic_rigidity": 0.52, "tribal_identification": 0.76, "temporal_orientation": "future"}},
    
    "Jacksonville": {"episodic": {"defining_moments": [{"event": "1999 AFC Championship runs", "emotional_weight": 0.75, "invoked_when": ["glory_days"], "timestamp": "1996-1999"}]}, "semantic": {"team_archetypes": {"Titans": "division_foe"}, "player_narratives": {"Trevor Lawrence": "hopeful_savior"}}, "procedural": {"argument_patterns": ["Hopeful about future"]}, "cognitive_dimensions": {"emotional_arousal": 0.68, "epistemic_rigidity": 0.48, "tribal_identification": 0.74, "temporal_orientation": "future"}},
    
    "Los Angeles Rams": {"episodic": {"defining_moments": [{"event": "Super Bowl LVI victory", "emotional_weight": 0.92, "invoked_when": ["championship", "Stafford"], "timestamp": "2022"}]}, "semantic": {"team_archetypes": {"49ers": "division_rival"}, "player_narratives": {"Matthew Stafford": "ring_finally"}}, "procedural": {"argument_patterns": ["Cite recent championship"]}, "cognitive_dimensions": {"emotional_arousal": 0.71, "epistemic_rigidity": 0.54, "tribal_identification": 0.79, "temporal_orientation": "present"}},
    
    "Detroit": {"episodic": {"defining_moments": [{"event": "0-16 season and perpetual losing", "emotional_weight": 0.85, "invoked_when": ["suffering", "cursed"], "timestamp": "2008"}]}, "semantic": {"team_archetypes": {"Packers": "division_dominators"}, "player_narratives": {"Barry Sanders": "wasted_legend"}}, "procedural": {"argument_patterns": ["Reference suffering"]}, "cognitive_dimensions": {"emotional_arousal": 0.74, "epistemic_rigidity": 0.50, "tribal_identification": 0.86, "temporal_orientation": "present"}},
    
    "Carolina": {"episodic": {"defining_moments": [{"event": "15-1 season, Super Bowl 50 loss", "emotional_weight": 0.82, "invoked_when": ["Cam", "heartbreak"], "timestamp": "2016"}]}, "semantic": {"team_archetypes": {"Saints": "division_rival"}, "player_narratives": {"Cam Newton": "mvp_era"}}, "procedural": {"argument_patterns": ["Reference 2015 season"]}, "cognitive_dimensions": {"emotional_arousal": 0.70, "epistemic_rigidity": 0.56, "tribal_identification": 0.76, "temporal_orientation": "present"}},
    
    "Arizona": {"episodic": {"defining_moments": [{"event": "Super Bowl XLIII - So close", "emotional_weight": 0.80, "invoked_when": ["heartbreak", "Steelers"], "timestamp": "2009"}]}, "semantic": {"team_archetypes": {"Seahawks": "division_foe"}, "player_narratives": {"Larry Fitzgerald": "loyal_legend"}}, "procedural": {"argument_patterns": ["Reference desert resilience"]}, "cognitive_dimensions": {"emotional_arousal": 0.68, "epistemic_rigidity": 0.54, "tribal_identification": 0.74, "temporal_orientation": "present"}},
    
    "Chicago": {"episodic": {"defining_moments": [{"event": "1985 Bears - Greatest defense", "emotional_weight": 0.94, "invoked_when": ["defense", "greatest", "85"], "timestamp": "1985"}]}, "semantic": {"team_archetypes": {"Packers": "historic_rival"}, "player_narratives": {"Walter Payton": "sweetness"}}, "procedural": {"argument_patterns": ["Invoke 85 Bears"]}, "cognitive_dimensions": {"emotional_arousal": 0.76, "epistemic_rigidity": 0.72, "tribal_identification": 0.88, "temporal_orientation": "past"}},
    
    "New York Jets": {"episodic": {"defining_moments": [{"event": "Namath's guarantee - Super Bowl III", "emotional_weight": 0.90, "invoked_when": ["championship", "Namath"], "timestamp": "1969"}]}, "semantic": {"team_archetypes": {"Patriots": "division_tormentors"}, "player_narratives": {"Joe Namath": "legend"}}, "procedural": {"argument_patterns": ["Reference 1969"]}, "cognitive_dimensions": {"emotional_arousal": 0.73, "epistemic_rigidity": 0.62, "tribal_identification": 0.82, "temporal_orientation": "past"}},
    
    "Washington": {"episodic": {"defining_moments": [{"event": "Hogs era - Three Super Bowls", "emotional_weight": 0.88, "invoked_when": ["championship", "80s"], "timestamp": "1982-1992"}]}, "semantic": {"team_archetypes": {"Cowboys": "division_rival"}, "player_narratives": {"Joe Gibbs": "coaching_legend"}}, "procedural": {"argument_patterns": ["Reference glory days"]}, "cognitive_dimensions": {"emotional_arousal": 0.70, "epistemic_rigidity": 0.64, "tribal_identification": 0.80, "temporal_orientation": "past"}}
}


def main():
    config_path = "config/city_profiles.json"
    
    # Load
    with open(config_path, 'r') as f:
        profiles = json.load(f)
    
    extended = 0
    for city, data in profiles.items():
        # Skip Philadelphia (already done)
        if city == "Philadelphia":
            continue
            
        if city in COMPLETE_P0_DATA:
            p0 = COMPLETE_P0_DATA[city]
            
            # Add to memory
            data["memory"]["episodic"] = p0["episodic"]
            data["memory"]["semantic"] = p0["semantic"]
            data["memory"]["procedural"] = p0["procedural"]
            
            # Add cognitive dimensions
            data["cognitive_dimensions"] = p0["cognitive_dimensions"]
            
            extended += 1
            print(f"✓ {city}")
    
    # Write
    with open(config_path, 'w') as f:
        json.dump(profiles, f, indent=4)
    
    print(f"\n✅ Extended {extended} cities (Philadelphia already had P0)")

if __name__ == "__main__":
    main()
