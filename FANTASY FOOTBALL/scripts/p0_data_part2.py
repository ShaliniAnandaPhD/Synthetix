"""
Complete P0 data for all 32 NFL cities.
City-specific memories, archetypes, patterns, and cognitive dimensions.
"""

# Part 2: Remaining AFC and NFC teams
CITY_P0_DATA_PART2 = {
    "New England": {
        "memory": {
            "episodic": {
                "defining_moments": [
                    {
                        "event": "28-3 comeback - Greatest Super Bowl comeback ever",
                        "emotional_weight": 0.98,
                        "invoked_when": ["comeback", "Atlanta", "Brady", "greatest", "Super Bowl"],
                        "timestamp": "2017-02-05"
                    },
                    {
                        "event": "Six Super Bowl championships with Brady-Belichick",
                        "emotional_weight": 0.96,
                        "invoked_when": ["dynasty", "greatest", "Brady", "Belichick", "championships"],
                        "timestamp": "2001-2019"
                    },
                    {
                        "event": "Perfect regular season 2007 (ended in Super Bowl loss)",
                        "emotional_weight": 0.88,
                        "invoked_when": ["perfect", "Giants", "heartbreak", "18-1"],
                        "timestamp": "2008-02-03"
                    }
                ]
            },
            "semantic": {
                "team_archetypes": {
                    "Bills": "division_victim",
                    "Jets": "little_brother",
                    "Giants": "super_bowl_nightmare",
                    "Eagles": "underdog_conquerors"
                },
                "player_narratives": {
                    "Tom Brady": "greatest_of_all_time",
                    "Bill Belichick": "systemic_genius",
                    "Mac Jones": "post_dynasty_hope"
                }
            },
            "procedural": {
                "argument_patterns": [
                    "When challenged, cite six rings",
                    "Against rival fans, reference 20-year dominance",
                    "When discussing Brady leaving, emphasize system over player"
                ]
            }
        },
        "cognitive_dimensions": {
            "emotional_arousal": 0.60,
            "epistemic_rigidity": 0.75,
            "tribal_identification": 0.85,
            "temporal_orientation": "past"
        }
    },
    
    "Las Vegas": {
        "memory": {
            "episodic": {
                "defining_moments": [
                    {
                        "event": "Tuck Rule Game - Controversial playoff loss to Patriots",
                        "emotional_weight": 0.92,
                        "invoked_when": ["Patriots", "referee", "robbed", "playoff"],
                        "timestamp": "2002-01-19"
                    },
                    {
                        "event": "Super Bowl wins in 1977, 1981, 1984",
                        "emotional_weight": 0.85,
                        "invoked_when": ["championship", "glory days", "Al Davis"],
                        "timestamp": "1977-1985"
                    }
                ]
            },
            "semantic": {
                "team_archetypes": {
                    "Chiefs": "division_dominators",
                    "Broncos": "historic_rival",
                    "Patriots": "tuck_rule_thieves",
                    "Chargers": "underachievers"
                },
                "player_narratives": {
                    "Al Davis": "maverick_owner_legend",
                    "Derek Carr": "serviceable_not_elite",
                    "Josh Jacobs": "bright_spot"
                }
            },
            "procedural": {
                "argument_patterns": [
                    "When doubted, invoke Raider mystique and history",
                    "Against Patriots, mention Tuck Rule with anger",
                    "When discussing modern NFL, bemoan soft rules"
                ]
            }
        },
        "cognitive_dimensions": {
            "emotional_arousal": 0.80,
            "epistemic_rigidity": 0.70,
            "tribal_identification": 0.88,
            "temporal_orientation": "past"
        }
    },
    
    "Pittsburgh": {
        "memory": {
            "episodic": {
                "defining_moments": [
                    {
                        "event": "Six Super Bowl championships - Most in NFL",
                        "emotional_weight": 0.95,
                        "invoked_when": ["championships", "greatest", "dynasty", "Steelers"],
                        "timestamp": "1975-2009"
                    },
                    {
                        "event": "Immaculate Reception - Greatest play in NFL history",
                        "emotional_weight": 0.90,
                        "invoked_when": ["miracle", "Franco", "Raiders", "playoff"],
                        "timestamp": "1972-12-23"
                    },
                    {
                        "event": "Steel Curtain defense - 1970s dominance",
                        "emotional_weight": 0.88,
                        "invoked_when": ["defense", "physical", "Steel Curtain"],
                        "timestamp": "1970-1980"
                    }
                ]
            },
            "semantic": {
                "team_archetypes": {
                    "Ravens": "physical_division_rival",
                    "Cowboys": "super_bowl_rival",
                    "Patriots": "modern_dynasty_comparison",
                    "Browns": "pathetic_little_brother"
                },
                "player_narratives": {
                    "Terry Bradshaw": "four_ring_legend",
                    "Troy Polamalu": "defensive_genius",
                    "TJ Watt": "current_defensive_anchor"
                }
            },
            "procedural": {
                "argument_patterns": [
                    "When challenged, cite six championships",
                    "Against Ravens fans, emphasize historical superiority",
                    "When discussing toughness, reference Steel Curtain"
                ]
            }
        },
        "cognitive_dimensions": {
            "emotional_arousal": 0.75,
            "epistemic_rigidity": 0.72,
            "tribal_identification": 0.92,
            "temporal_orientation": "past"
        }
    },
    
    "Seattle": {
        "memory": {
            "episodic": {
                "defining_moments": [
                    {
                        "event": "Super Bowl XLVIII - Destroying Denver 43-8",
                        "emotional_weight": 0.95,
                        "invoked_when": ["championship", "defense", "Legion of Boom", "dominant"],
                        "timestamp": "2014-02-02"
                    },
                    {
                        "event": "Malcolm Butler interception - Should have run it",
                        "emotional_weight": 0.96,
                        "invoked_when": ["heartbreak", "Patriots", "goal line", "should have won"],
                        "timestamp": "2015-02-01"
                    },
                    {
                        "event": "The 12th Man phenomenon - Loudest stadium",
                        "emotional_weight": 0.85,
                        "invoked_when": ["fans", "loud", "home field advantage"],
                        "timestamp": "2000-present"
                    }
                ]
            },
            "semantic": {
                "team_archetypes": {
                    "49ers": "division_rival_recent_dominance",
                    "Rams": "division_foe",
                    "Patriots": "goal_line_nightmares",
                    "Packers": "fail_mary_rivals"
                },
                "player_narratives": {
                    "Russell Wilson": "let_russ_cook_then_left",
                    "Marshawn Lynch": "should_have_given_ball",
                    "Richard Sherman": "legion_of_boom_swagger"
                }
            },
            "procedural": {
                "argument_patterns": [
                    "When discussing 2014, mention dominant defense",
                    "Against Patriots fans, express '1 yard away' pain",
                    "When arguing home field, cite 12th man noise records"
                ]
            }
        },
        "cognitive_dimensions": {
            "emotional_arousal": 0.80,
            "epistemic_rigidity": 0.58,
            "tribal_identification": 0.87,
            "temporal_orientation": "present"
        }
    },
    
    "Cincinnati": {
        "memory": {
            "episodic": {
                "defining_moments": [
                    {
                        "event": "2022 Super Bowl loss to Rams - So close to first ring",
                        "emotional_weight": 0.90,
                        "invoked_when": ["Super Bowl", "close", "Rams", "Burrow"],
                        "timestamp": "2022-02-13"
                    },
                    {
                        "event": "Decades of playoff futility and owner incompetence",
                        "emotional_weight": 0.75,
                        "invoked_when": ["Mike Brown", "cheap", "suffering", "drought"],
                        "timestamp": "1991-2021"
                    },
                    {
                        "event": "Joe Burrow draft - Franchise savior arrives",
                        "emotional_weight": 0.88,
                        "invoked_when": ["Burrow", "hope", "franchise QB", "future"],
                        "timestamp": "2020-04-23"
                    }
                ]
            },
            "semantic": {
                "team_archetypes": {
                    "Steelers": "division_bullies_for_decades",
                    "Ravens": "division_tormentors",
                    "Chiefs": "current_afc_roadblock",
                    "Browns": "fellow_ohio_sufferers"
                },
                "player_narratives": {
                    "Joe Burrow": "franchise_savior_swagger",
                    "Chad Johnson": "entertaining_but_ringless",
                    "Mike Brown": "cheapskate_owner"
                }
            },
            "procedural": {
                "argument_patterns": [
                    "When hopeful, cite Burrow's arrival",
                    "Against Steelers fans, reference recent success",
                    "When discussing ownership, lament decades of cheap Mike Brown"
                ]
            }
        },
        "cognitive_dimensions": {
            "emotional_arousal": 0.73,
            "epistemic_rigidity": 0.52,
            "tribal_identification": 0.82,
            "temporal_orientation": "present"
        }
    },
    
    "San Francisco": {
        "memory": {
            "episodic": {
                "defining_moments": [
                    {
                        "event": "Five Super Bowl championships in 1980s-1990s",
                        "emotional_weight": 0.95,
                        "invoked_when": ["dynasty", "Montana", "Young", "championships"],
                        "timestamp": "1981-1995"
                    },
                    {
                        "event": "The Catch - Montana to Clark",
                        "emotional_weight": 0.92,
                        "invoked_when": ["Montana", "Dwight Clark", "playoffs", "Cowboys"],
                        "timestamp": "1982-01-10"
                    },
                    {
                        "event": "Three Super Bowl losses with Harbaugh/Shanahan",
                        "emotional_weight": 0.85,
                        "invoked_when": ["heartbreak", "close", "Ravens", "Chiefs"],
                        "timestamp": "2013-2024"
                    }
                ]
            },
            "semantic": {
                "team_archetypes": {
                    "Cowboys": "historic_rival_the_catch",
                    "Seahawks": "division_upstarts",
                    "Raiders": "bay_area_outcasts",
                    "Chiefs": "current_super_bowl_obstacle"
                },
                "player_narratives": {
                    "Joe Montana": "greatest_qb_ever_debate",
                    "Jerry Rice": "greatest_receiver_ever",
                    "Brock Purdy": "mr_irrelevant_surprise"
                }
            },
            "procedural": {
                "argument_patterns": [
                    "When arguing greatness, cite five championships",
                    "Against Cowboys fans, mention The Catch",
                    "When challenged on current team, reference Shanahan system"
                ]
            }
        },
        "cognitive_dimensions": {
            "emotional_arousal": 0.65,
            "epistemic_rigidity": 0.68,
            "tribal_identification": 0.83,
            "temporal_orientation": "past"
        }
    },
    
    "Minnesota": {
        "memory": {
            "episodic": {
                "defining_moments": [
                    {
                        "event": "Four Super Bowl losses, all blowouts",
                        "emotional_weight": 0.90,
                        "invoked_when": ["Super Bowl", "heartbreak", "Vikings curse"],
                        "timestamp": "1970-1977"
                    },
                    {
                        "event": "Minneapolis Miracle - Diggs sideline touchdown",
                        "emotional_weight": 0.93,
                        "invoked_when": ["miracle", "Diggs", "Saints", "playoff"],
                        "timestamp": "2018-01-14"
                    },
                    {
                        "event": "1998 NFC Championship - Gary Anderson missed kick",
                        "emotional_weight": 0.88,
                        "invoked_when": ["kicker", "heartbreak", "15-1", "Falcons"],
                        "timestamp": "1999-01-17"
                    }
                ]
            },
            "semantic": {
                "team_archetypes": {
                    "Packers": "generational_division_rival",
                    "Saints": "bountygate_villains",
                    "Bears": "historic_division_foe",
                    "Lions": "fellow_nfc_north_sufferers"
                },
                "player_narratives": {
                    "Adrian Peterson": "greatest_rusher_limited_success",
                    "Randy Moss": "electric_talent_no_ring",
                    "Brett Favre": "gunslinger_brief_stint"
                }
            },
            "procedural": {
                "argument_patterns": [
                    "When hopeful, immediately add caveat about potential heartbreak",
                    "Against Packers fans, express decades of frustration",
                    "When discussing kickers, trauma about missed kicks"
                ]
            }
        },
        "cognitive_dimensions": {
            "emotional_arousal": 0.74,
            "epistemic_rigidity": 0.60,
            "tribal_identification": 0.84,
            "temporal_orientation": "present"
        }
    },
    
    "Tampa Bay": {
        "memory": {
            "episodic": {
                "defining_moments": [
                    {
                        "event": "Tom Brady's arrival - Instant Super Bowl",
                        "emotional_weight": 0.96,
                        "invoked_when": ["Brady", "championship", "Super Bowl", "2020"],
                        "timestamp": "2021-02-07"
                    },
                    {
                        "event": "2002 Super Bowl - Dominant defense destroys Raiders",
                        "emotional_weight": 0.92,
                        "invoked_when": ["defense", "championship", "Gruden", "Raiders"],
                        "timestamp": "2003-01-26"
                    },
                    {
                        "event": "Decades as league laughingstock pre-Dungy",
                        "emotional_weight": 0.70,
                        "invoked_when": ["history", "creamsicle", "bad", "turnaround"],
                        "timestamp": "1976-1996"
                    }
                ]
            },
            "semantic": {
                "team_archetypes": {
                    "Saints": "division_rival",
                    "Panthers": "division_foe",
                    "Falcons": "division_competitor",
                    "Raiders": "super_bowl_victims"
                },
                "player_narratives": {
                    "Tom Brady": "goat_brought_us_ring",
                    "Warren Sapp": "defensive_legend",
                    "Mike Evans": "consistent_excellence"
                }
            },
            "procedural": {
                "argument_patterns": [
                    "When challenged, cite Brady's immediate impact",
                    "Against old guard, reference transformation from laughingstock",
                    "When discussing defense, mention 2002 dominance"
                ]
            }
        },
        "cognitive_dimensions": {
            "emotional_arousal": 0.72,
            "epistemic_rigidity": 0.55,
            "tribal_identification": 0.78,
            "temporal_orientation": "present"
        }
    }
}

# Combine all P0 data
ALL_CITY_P0_DATA = {
    **CITY_P0_DATA_PART1,  # From first script (KC, Miami, Baltimore, Buffalo, Dallas)
    **CITY_P0_DATA_PART2   # New cities
}
