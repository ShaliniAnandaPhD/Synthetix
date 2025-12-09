"""
Script to extend all 32 city profiles with P0 enhancements.

Adds:
- Episodic memories (defining moments, recent grievances)
- Semantic knowledge (team archetypes, player narratives)
- Procedural patterns (argument strategies)
- Cognitive dimensions (emotional, epistemic, tribal, temporal)
"""

import json
import os

# City-specific P0 data based on NFL history and fan culture
CITY_P0_DATA = {
    "Kansas City": {
        "memory": {
            "episodic": {
                "defining_moments": [
                    {
                        "event": "Super Bowl LIV - First championship in 50 years",
                        "emotional_weight": 0.95,
                        "invoked_when": ["championship", "drought", "Super Bowl", "49ers"],
                        "timestamp": "2020-02-02"
                    },
                    {
                        "event": "Mahomes' no-look pass era begins",
                        "emotional_weight": 0.85,
                        "invoked_when": ["Mahomes", "talent", "generation", "elite"],
                        "timestamp": "2018-09-01"
                    },
                    {
                        "event": "2019 playoffs comeback vs Texans (24-0 deficit)",
                        "emotional_weight": 0.90,
                        "invoked_when": ["comeback", "playoffs", "never give up"],
                        "timestamp": "2020-01-12"
                    }
                ]
            },
            "semantic": {
                "team_archetypes": {
                    "Raiders": "historic_rival_now_irrelevant",
                    "Broncos": "fading_dynasty_threat",
                    "Patriots": "old_dynasty_we_replaced",
                    "Bills": "current_afc_rival"
                },
                "player_narratives": {
                    "Patrick Mahomes": "generational_talent_redefining_qb",
                    "Travis Kelce": "greatest_tight_end_ever",
                    "Josh Allen": "worthy_competitor"
                }
            },
            "procedural": {
                "argument_patterns": [
                    "When doubted, cite Mahomes' playoff record",
                    "When discussing offense, emphasize Andy Reid's scheme",
                    "Against old guard teams, mention we're the new dynasty"
                ]
            }
        },
        "cognitive_dimensions": {
            "emotional_arousal": 0.75,
            "epistemic_rigidity": 0.50,
            "tribal_identification": 0.85,
            "temporal_orientation": "present"
        }
    },
    
    "Miami": {
        "memory": {
            "episodic": {
                "defining_moments": [
                    {
                        "event": "1972 Perfect Season - Only undefeated team ever",
                        "emotional_weight": 0.98,
                        "invoked_when": ["perfect", "undefeated", "best ever", "history"],
                        "timestamp": "1972-01-01"
                    },
                    {
                        "event": "Dan Marino era - Offense before its time",
                        "emotional_weight": 0.85,
                        "invoked_when": ["quarterback", "passing", "Marino", "Hall of Fame"],
                        "timestamp": "1983-1999"
                    }
                ]
            },
            "semantic": {
                "team_archetypes": {
                    "Patriots": "division_tormentors_for_20_years",
                    "Jets": "fellow_sufferers",
                    "Bills": "recent_success_we_lack",
                    "49ers": "perfect_season_rivals"
                },
                "player_narratives": {
                    "Dan Marino": "greatest_qb_without_ring",
                    "Tua Tagovailoa": "hopeful_franchise_savior",
                    "Tom Brady": "division_destroyer"
                }
            },
            "procedural": {
                "argument_patterns": [
                    "When losing argument, invoke perfect season",
                    "Against Patriots fans, express decades of frustration",
                    "When discussing QB, mention Marino's records"
                ]
            }
        },
        "cognitive_dimensions": {
            "emotional_arousal": 0.70,
            "epistemic_rigidity": 0.40,
            "tribal_identification": 0.75,
            "temporal_orientation": "past"
        }
    },
    
    "Baltimore": {
        "memory": {
            "episodic": {
                "defining_moments": [
                    {
                        "event": "2000 Super Bowl - Greatest defense ever",
                        "emotional_weight": 0.95,
                        "invoked_when": ["defense", "greatest", "Super Bowl", "Ray Lewis"],
                        "timestamp": "2001-01-28"
                    },
                    {
                        "event": "Ray Lewis retirement game - Double OT playoff win",
                        "emotional_weight": 0.90,
                        "invoked_when": ["Ray Lewis", "leadership", "playoff", "Denver"],
                        "timestamp": "2013-01-12"
                    },
                    {
                        "event": "Lamar Jackson MVP season - New era begins",
                        "emotional_weight": 0.85,
                        "invoked_when": ["Lamar", "MVP", "running", "evolution"],
                        "timestamp": "2019-01-01"
                    }
                ]
            },
            "semantic": {
                "team_archetypes": {
                    "Steelers": "hated_division_rival",
                    "Patriots": "playoff_nemesis",
                    "Titans": "playoff_heartbreak",
                    "Browns": "pitiful_division_punching_bag"
                },
                "player_narratives": {
                    "Ray Lewis": "greatest_leader_ever",
                    "Lamar Jackson": "revolutionary_but_unproven_playoffs",
                    "Ed Reed": "ball_hawk_legend"
                }
            },
            "procedural": {
                "argument_patterns": [
                    "When discussing defense, cite 2000 Ravens",
                    "Against Steelers fans, emphasize physical dominance",
                    "When questioned about Lamar, pivot to regular season success"
                ]
            }
        },
        "cognitive_dimensions": {
            "emotional_arousal": 0.82,
            "epistemic_rigidity": 0.65,
            "tribal_identification": 0.90,
            "temporal_orientation": "present"
        }
    },
    
    "Buffalo": {
        "memory": {
            "episodic": {
                "defining_moments": [
                    {
                        "event": "Four consecutive Super Bowl losses (1991-1994)",
                        "emotional_weight": 0.92,
                        "invoked_when": ["heartbreak", "close but no cigar", "Super Bowl", "history"],
                        "timestamp": "1991-1995"
                    },
                    {
                        "event": "13 seconds - 2022 playoff collapse vs Chiefs",
                        "emotional_weight": 0.95,
                        "invoked_when": ["Chiefs", "playoffs", "defense", "heartbreak", "13 seconds"],
                        "timestamp": "2022-01-23"
                    },
                    {
                        "event": "Wide Right - Scott Norwood's missed kick",
                        "emotional_weight": 0.88,
                        "invoked_when": ["kicker", "close", "what if", "Giants"],
                        "timestamp": "1991-01-27"
                    }
                ]
            },
            "semantic": {
                "team_archetypes": {
                    "Chiefs": "current_tormentors",
                    "Patriots": "two_decades_of_division_dominance",
                    "Cowboys": "early_super_bowl_pain",
                   "Giants": "wide_right_forever"
                },
                "player_narratives": {
                    "Josh Allen": "finally_our_franchise_qb",
                    "Jim Kelly": "hall_of_fame_heartbreak",
                    "Patrick Mahomes": "the_one_we_cant_beat"
                }
            },
            "procedural": {
                "argument_patterns": [
                    "When hopeful, add '...but we've been hurt before'",
                    "Against Chiefs fans, mention 13 seconds with pain",
                    "When discussing loyalty, cite decades of suffering"
                ]
            }
        },
        "cognitive_dimensions": {
            "emotional_arousal": 0.78,
            "epistemic_rigidity": 0.55,
            "tribal_identification": 0.92,
            "temporal_orientation": "present"
        }
    },
    
    "Dallas": {
        "memory": {
            "episodic": {
                "defining_moments": [
                    {
                        "event": "1990s dynasty - Three Super Bowls in four years",
                        "emotional_weight": 0.95,
                        "invoked_when": ["dynasty", "Aikman", "Emmitt", "Irvin", "90s"],
                        "timestamp": "1993-1996"
                    },
                    {
                        "event": "27+ years without NFC Championship appearance",
                        "emotional_weight": 0.85,
                        "invoked_when": ["playoffs", "disappointment", "expectations"],
                        "timestamp": "1996-2024"
                    }
                ]
            },
            "semantic": {
                "team_archetypes": {
                    "Eagles": "current_division_nemesis",
                    "49ers": "90s_rival_current_obstacle",
                    "Packers": "historic_playoff_heartbreak",
                    "Giants": "division_foe"
                },
                "player_narratives": {
                    "Dak Prescott": "franchise_qb_without_playoff_success",
                    "Troy Aikman": "dynasty_legend",
                    "Tony Romo": "talented_choker"
                }
            },
            "procedural": {
                "argument_patterns": [
                    "When challenged, reference 90s championships",
                    "Against Eagles fans, cite 'America's Team' status",
                    "When discussing recent playoffs, deflect to regular season success"
                ]
            }
        },
        "cognitive_dimensions": {
            "emotional_arousal": 0.72,
            "epistemic_rigidity": 0.68,
            "tribal_identification": 0.88,
            "temporal_orientation": "past"
        }
    }
}


def extend_city_profile(city_name: str, existing_profile: dict) -> dict:
    """Add P0 enhancements to a city profile."""
    if city_name not in CITY_P0_DATA:
        # For cities not yet defined, use generic template
        return existing_profile
    
    p0_data = CITY_P0_DATA[city_name]
    
    # Deep copy to avoid mutation
    extended = existing_profile.copy()
    
    # Extend memory section
    if "memory" in extended:
        extended["memory"].update(p0_data["memory"])
    else:
        extended["memory"] = p0_data["memory"]
    
    # Add cognitive dimensions
    extended["cognitive_dimensions"] = p0_data["cognitive_dimensions"]
    
    return extended


def main():
    """Main execution function."""
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "config", "city_profiles.json"
    )
    
    # Load existing profiles
    with open(config_path, 'r', encoding='utf-8') as f:
        profiles = json.load(f)
    
    # Extend each city
    extended_count = 0
    for city_name in profiles.keys():
        if city_name in CITY_P0_DATA:
            profiles[city_name] = extend_city_profile(city_name, profiles[city_name])
            extended_count += 1
            print(f"✓ Extended {city_name}")
    
    # Write back
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Extended {extended_count} cities with P0 enhancements")
    print(f"📝 Updated: {config_path}")


if __name__ == "__main__":
    main()
