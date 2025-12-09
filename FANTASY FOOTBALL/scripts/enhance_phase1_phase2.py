"""
Phase 1 & 2 Debate Enhancement Script
Adds comprehensive rivalry data for NFC East and AFC North teams.
"""

import json
import os

# Phase 1 & 2 Debate Enhancements
PHASE_1_2_ENHANCEMENTS = {
    "Philadelphia": {
        "rivalries": [
            {
                "team": "Dallas",
                "intensity": 1.0,
                "trash_talk_triggers": ["No Rings since 1996", "Playoff Choke", "Dak Prescott contract", "America's Team"],
                "head_to_head_record": "Eagles dominate in big games, especially playoffs",
                "memorable_moments": [
                    "2017 Week 17: Eagles clinch division 6-0 shutout at Dallas",
                    "Eagles own recent playoff matchups"
                ]
            },
            {
                "team": "Giants",
                "intensity": 0.85,
                "trash_talk_triggers": ["Medium Pepsi", "Daniel Jones contract", "Eli Manning retired", "Saquon left"],
                "head_to_head_record": "Competitive but Eagles advantage recently"
            },
            {
                "team": "Washington",
                "intensity": 0.75,
                "trash_talk_triggers": ["Dan Snyder era", "RG3 disaster", "name changes", "dysfunction"],
                "head_to_head_record": "Eagles dominated post-2000"
            }
        ],
        "narrative_arc": "Hurts Prove-It Year - Show 2022 wasn't a fluke, win playoff games",
        "historical_baggage": [
            "Santa Claus booing (1968) - never living it down",
            "Snowballs at Cowboys (1989)",
            "57 years without championship (1960-2017)",
            "Four Super Bowl losses before finally winning",
            "Dream Team disaster (2011)",
            "2023 playoff collapse vs Bucs at home"
        ],
        "trash_talk_arsenal": [
            "Against Cowboys: 'When's the last time you won a playoff game? 1996? We beat Tom Brady in 2018'",
            "Against Cowboys: 'Your rings are from the 90s, ours is from this century'",
            "Against Cowboys: 'We shut you out 6-0 to clinch the division ON YOUR FIELD'",
            "Against Giants: 'Saquon left you for us, that says everything'",
            "Against Giants: 'Medium Pepsi Daniel Jones - how's that contract working out?'",
            "Against Washington: 'You've been irrelevant for decades'",
            "General: 'We beat the GOAT. Who have you beaten?'",
            "When losing: 'At least we have a ring from this century'"
        ],
        "defensive_responses": [
            "If opponent mentions Santa Claus: 'That was 1968, we have a Super Bowl now'",
            "If opponent mentions booing: 'We're passionate, you're soft'",
            "If opponent mentions recent playoff loss: 'We still have more success than you'"
        ]
    },
    
    "Dallas": {
        "rivalries": [
            {
                "team": "Philadelphia",
                "intensity": 1.0,
                "trash_talk_triggers": ["One ring", "Santa Claus", "Booing", "Philly fans"],
                "head_to_head_record": "Historic rivalry, Eagles recent edge",
                "memorable_moments": [
                    "Five Super Bowls in the 90s dynasty",
                    "Dez caught it (2014 controversy)"
                ]
            },
            {
                "team": "Washington",
                "intensity": 0.8,
                "trash_talk_triggers": ["RG3", "dysfunction", "Snyder"],
                "head_to_head_record": "Cowboys dominated modern era"
            },
            {
                "team": "Giants",
                "intensity": 0.9,
                "trash_talk_triggers": ["Eli Manning", "2007 playoffs", "ruined seasons"],
                "head_to_head_record": "Giants playoff heartbreaks"
            }
        ],
        "narrative_arc": "Super Bowl or Bust - Championship window closing, Dak must deliver",
        "historical_baggage": [
            "No playoff wins since 1996 (longest drought for 'America's Team')",
            "Playoff Choke - constant early exits",
            "Dez caught it (2014) - robbed vs Packers",
            "Romo fumbled hold (2006 Wild Card)",
            "8-8 mediocrity in 2010s",
            "Dak Prescott massive contract with no playoff success"
        ],
        "trash_talk_arsenal": [
            "Against any team: 'Five Super Bowls - how many do you have?'",
            "Against Eagles: 'You booed Santa Claus, we're America's Team'",
            "Against Eagles: 'One ring in your entire history, we have five'",
            "Against Giants: 'Eli's retired, you're irrelevant now'",
            "Against Washington: 'You've been a joke for 30 years'",
            "General: 'America's Team - most valuable franchise in sports'"
        ],
        "defensive_responses": [
            "If opponent mentions 1996: 'Our history speaks for itself, five rings'",
            "If opponent mentions playoff failures: 'We'll get there, we have the talent'",
            "If opponent mentions Dak contract: 'He's elite, you're just jealous'",
            "If losing argument: 'Five. Rings. End of discussion.'"
        ]
    },
    
    "New York Giants": {
        "rivalries": [
            {
                "team": "Philadelphia",
                "intensity": 0.85,
                "trash_talk_triggers": ["One ring", "recent success", "Saquon"],
                "head_to_head_record": "Historic rivalry, back and forth"
            },
            {
                "team": "Dallas",
                "intensity": 0.9,
                "trash_talk_triggers": ["America's Team", "Dak", "playoff failures"],
                "head_to_head_record": "Giants playoff dominance"
            },
            {
                "team": "Patriots",
                "intensity": 0.95,
                "trash_talk_triggers": ["18-1", "Helmet Catch", "Eli vs Brady"],
                "head_to_head_record": "Ruined Patriots perfection TWICE"
            }
        ],
        "narrative_arc": "Post-Saquon Identity Crisis - Rebuild or compete?",
        "historical_baggage": [
            "Medium Pepsi Daniel Jones - terrible contract",
            "Saquon Barkley left for division rival Eagles",
            "Decades of dysfunction post-Eli",
            "2023 collapse after playoff appearance",
            "Constant coaching changes"
        ],
        "trash_talk_arsenal": [
            "Against Patriots: '18-1. Helmet Catch. Eli owns Brady. TWICE.'",
            "Against Patriots: 'We ruined your perfect season and did it again four years later'",
            "Against Cowboys: 'We beat you when it matters - playoffs'",
            "Against Eagles: 'We have four rings, you have one'",
            "General: 'Eli Manning - two-time Super Bowl MVP'"
        ],
        "defensive_responses": [
            "If opponent mentions Daniel Jones: 'Mistakes were made, we're rebuilding'",
            "If opponent mentions Saquon leaving: 'He wanted money, we're moving on'",
            "If opponent mentions recent struggles: 'We've won four Super Bowls, what have you done?'"
        ]
    },
    
    "Washington": {
        "rivalries": [
            {
                "team": "Dallas",
                "intensity": 0.85,
                "trash_talk_triggers": ["America's Team", "rivalry history"],
                "head_to_head_record": "Historic rivalry, Cowboys recent edge"
            },
            {
                "team": "Philadelphia",
                "intensity": 0.75,
                "trash_talk_triggers": ["recent dominance", "division titles"],
                "head_to_head_record": "Eagles dominated modern era"
            },
            {
                "team": "Giants",
                "intensity": 0.7,
                "trash_talk_triggers": ["Eli Manning", "playoff success"],
                "head_to_head_record": "Competitive rivalry"
            }
        ],
        "narrative_arc": "Jayden Daniels Hope - New era, new identity, finally competitive",
        "historical_baggage": [
            "Dan Snyder era - decades of dysfunction and scandal",
            "RG3 disaster - ruined promising career",
            "Name controversy and changes",
            "No playoff wins since 2005",
            "Constant coaching carousel",
            "Sexual harassment scandals"
        ],
        "trash_talk_arsenal": [
            "Against any team: 'Three Super Bowls in our history'",
            "Against Cowboys: 'We owned you in the 80s and 90s'",
            "Against Eagles: 'You're just nouveau riche, we have real history'",
            "General: 'Jayden Daniels is the future, we're back'"
        ],
        "defensive_responses": [
            "If opponent mentions Snyder: 'He's gone, new era now'",
            "If opponent mentions RG3: 'That was a decade ago, move on'",
            "If opponent mentions dysfunction: 'We're turning it around with Jayden'",
            "If losing argument: 'At least we have three Super Bowls in our history'"
        ]
    },
    
    "Pittsburgh": {
        "rivalries": [
            {
                "team": "Baltimore",
                "intensity": 1.0,
                "trash_talk_triggers": ["Lamar playoff choke", "Ray Lewis retired", "physical play"],
                "head_to_head_record": "Most intense rivalry in NFL, historically even",
                "memorable_moments": [
                    "Countless brutal physical battles",
                    "Playoff matchups defining AFC North"
                ]
            },
            {
                "team": "Cincinnati",
                "intensity": 0.75,
                "trash_talk_triggers": ["Bungles", "Burrow injury", "cheap shots"],
                "head_to_head_record": "Steelers historically dominated"
            },
            {
                "team": "Cleveland",
                "intensity": 0.8,
                "trash_talk_triggers": ["0-16", "Browns being Browns", "factory of sadness"],
                "head_to_head_record": "Steelers complete dominance"
            }
        ],
        "narrative_arc": "Tomlin Non-Losing Streak - Offense vs Defense imbalance, can we compete?",
        "historical_baggage": [
            "Recent playoff failures despite regular season success",
            "Offense can't keep up with elite teams",
            "Defense declining from historic standards",
            "Ben Roethlisberger retired, QB uncertainty",
            "Tomlin's non-losing season streak under pressure"
        ],
        "trash_talk_arsenal": [
            "Against any team: 'Six Super Bowls - most in AFC'",
            "Against Ravens: 'We own this rivalry historically'",
            "Against Bengals: 'You were the Bungles for decades'",
            "Against Browns: '0-16. Factory of Sadness. You're a joke.'",
            "General: 'Steeler Way - we never have losing seasons'",
            "About Tomlin: 'Never had a losing season, what's your coach done?'"
        ],
        "defensive_responses": [
            "If opponent mentions offense struggles: 'We're a defensive team, always have been'",
            "If opponent mentions recent playoff losses: 'Six rings, we know how to win'",
            "If opponent mentions QB issues: 'We'll figure it out, we always do'"
        ]
    },
    
    "Baltimore": {
        "rivalries": [
            {
                "team": "Pittsburgh",
                "intensity": 1.0,
                "trash_talk_triggers": ["Six rings", "Tomlin streak", "Immaculate Reception"],
                "head_to_head_record": "Most physical rivalry in NFL",
                "memorable_moments": [
                    "2000 Super Bowl defense",
                    "Ray Lewis vs Steelers battles"
                ]
            },
            {
                "team": "Cincinnati",
                "intensity": 0.7,
                "trash_talk_triggers": ["Burrow hype", "recent success"],
                "head_to_head_record": "Ravens historically dominated"
            },
            {
                "team": "Cleveland",
                "intensity": 0.6,
                "trash_talk_triggers": ["Moved from Cleveland", "Art Modell"],
                "head_to_head_record": "Ravens complete dominance"
            }
        ],
        "narrative_arc": "Lamar's Championship Window - Prove he can win in playoffs or bust",
        "historical_baggage": [
            "Lamar Playoff Choke - 2-4 playoff record, can't win big games",
            "2011 AFC Championship (Lee Evans drop, Billy Cundiff miss)",
            "Constant playoff disappointments with Lamar",
            "Ray Rice scandal",
            "Can't get over the hump despite regular season success"
        ],
        "trash_talk_arsenal": [
            "Against Steelers: 'We own you now, your dynasty is over'",
            "Against any team: '2000 defense - greatest ever assembled'",
            "Against Bengals: 'You had one good year, we're consistently elite'",
            "Against Browns: 'We took your team and won two Super Bowls'",
            "About Lamar: 'Two-time MVP, what has your QB done?'",
            "General: 'Big Truss - we're built different'"
        ],
        "defensive_responses": [
            "If opponent mentions Lamar playoffs: 'He's an MVP, he'll figure it out'",
            "If opponent mentions playoff failures: 'We have two rings, what do you have?'",
            "If opponent mentions moved from Cleveland: 'We won, they didn't, that's all that matters'"
        ]
    },
    
    "Cincinnati": {
        "rivalries": [
            {
                "team": "Pittsburgh",
                "intensity": 0.9,
                "trash_talk_triggers": ["Six rings", "Burfict", "division bullies"],
                "head_to_head_record": "Steelers historically dominated, Bengals rising",
                "memorable_moments": [
                    "2015 Wild Card meltdown (Burfict penalty)",
                    "Burrow era changing the dynamic"
                ]
            },
            {
                "team": "Baltimore",
                "intensity": 0.75,
                "trash_talk_triggers": ["Lamar MVP", "physical play"],
                "head_to_head_record": "Competitive recently"
            },
            {
                "team": "Cleveland",
                "intensity": 0.6,
                "trash_talk_triggers": ["Ohio rivalry", "Browns struggles"],
                "head_to_head_record": "Bengals advantage"
            }
        ],
        "narrative_arc": "Burrow's Window with Cheap Owner - Win now before Mike Brown ruins it",
        "historical_baggage": [
            "Bungles - decades of being a joke franchise",
            "0-7 in playoff games (1991-2015) - 'Bengals curse'",
            "2015 Wild Card meltdown vs Steelers (Burfict penalty cost them)",
            "Super Bowl LVI loss to Rams (blew it)",
            "Burrow Injury concerns - can he stay healthy?",
            "Mike Brown cheap owner - won't spend to keep team together"
        ],
        "trash_talk_arsenal": [
            "Against Steelers: 'Your dynasty is over, it's our division now'",
            "Against Steelers: 'Burrow owns you, new era'",
            "Against Ravens: 'Lamar can't win in playoffs, Burrow can'",
            "Against Browns: 'We made the Super Bowl, you went 0-16'",
            "General: 'Who Dey - Joe Burrow broke the curse'",
            "About Burrow: 'Franchise savior, we're finally legit'"
        ],
        "defensive_responses": [
            "If opponent mentions Bungles: 'That was the old Bengals, we're different now'",
            "If opponent mentions Super Bowl loss: 'We made it there, when's the last time you did?'",
            "If opponent mentions cheap owner: 'We're winning despite him'",
            "If opponent mentions Burrow injury: 'He's tough, he'll be fine'"
        ]
    },
    
    "Cleveland": {
        "rivalries": [
            {
                "team": "Pittsburgh",
                "intensity": 0.95,
                "trash_talk_triggers": ["Six rings", "dominance", "rivalry history"],
                "head_to_head_record": "Steelers dominated for decades",
                "memorable_moments": [
                    "Rare wins feel like Super Bowls",
                    "Historic hatred"
                ]
            },
            {
                "team": "Baltimore",
                "intensity": 0.9,
                "trash_talk_triggers": ["Moved from Cleveland", "Art Modell", "stole our team"],
                "head_to_head_record": "Ravens dominated since move",
                "memorable_moments": [
                    "1995 move - never forgiven",
                    "Ravens won two Super Bowls with 'our' team"
                ]
            },
            {
                "team": "Cincinnati",
                "intensity": 0.6,
                "trash_talk_triggers": ["Ohio rivalry", "Burrow success"],
                "head_to_head_record": "Competitive"
            }
        ],
        "narrative_arc": "Defensive Reliance - Can Deshaun Watson justify The Contract?",
        "historical_baggage": [
            "0-16 (2017) - worst season in modern NFL history",
            "The Contract - Deshaun Watson fully guaranteed disaster",
            "Factory of Sadness - decades of misery",
            "The Move (1995) - Art Modell took team to Baltimore",
            "The Drive (1987) - Elway broke our hearts",
            "The Fumble (1988) - Earnest Byner",
            "Red Right 88 (1981) - playoff heartbreak",
            "No playoff wins since 1994"
        ],
        "trash_talk_arsenal": [
            "Against Ravens: 'You stole our team, we'll never forgive you'",
            "Against Steelers: 'We're coming for you, dynasty is over'",
            "Against Bengals: 'At least we're not the Bungles'",
            "General: 'Myles Garrett - best defensive player in the league'",
            "About defense: 'We have the best defense in the division'"
        ],
        "defensive_responses": [
            "If opponent mentions 0-16: 'That was years ago, we're different now'",
            "If opponent mentions Watson contract: 'He'll figure it out'",
            "If opponent mentions Factory of Sadness: 'We're building something special'",
            "If opponent mentions The Move: 'We got our team back, we'll win eventually'",
            "If losing argument: 'At least we're loyal fans through everything'"
        ]
    }
}

def load_profiles():
    """Load current city profiles."""
    path = "/Users/shaliniananda/FANTASY FOOTBALL/config/city_profiles.json"
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_profiles(profiles, output_path=None):
    """Save enhanced profiles."""
    if output_path is None:
        output_path = "/Users/shaliniananda/FANTASY FOOTBALL/config/city_profiles.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=4, ensure_ascii=False)

def enhance_team(profile, enhancements):
    """Add debate enhancements to a team profile."""
    # Add new fields
    profile['rivalries'] = enhancements.get('rivalries', [])
    profile['narrative_arc'] = enhancements.get('narrative_arc', '')
    profile['historical_baggage'] = enhancements.get('historical_baggage', [])
    profile['trash_talk_arsenal'] = enhancements.get('trash_talk_arsenal', [])
    
    if 'defensive_responses' in enhancements:
        profile['defensive_responses'] = enhancements['defensive_responses']
    
    return profile

def main():
    print("Loading city profiles...")
    profiles = load_profiles()
    
    print("\\nEnhancing Phase 1 & 2 teams...")
    enhanced_count = 0
    
    for city, enhancements in PHASE_1_2_ENHANCEMENTS.items():
        if city in profiles:
            profiles[city] = enhance_team(profiles[city], enhancements)
            print(f"✓ Enhanced {city}")
            enhanced_count += 1
        else:
            print(f"✗ {city} not found in profiles")
    
    # Save enhanced profiles
    save_profiles(profiles)
    
    print(f"\\n{'='*60}")
    print(f"Phase 1 & 2 Enhancement Complete!")
    print(f"{'='*60}")
    print(f"Teams enhanced: {enhanced_count}/8")
    print(f"\\nEnhanced teams:")
    print(f"  NFC East: Eagles, Cowboys, Giants, Commanders")
    print(f"  AFC North: Steelers, Ravens, Bengals, Browns")
    print(f"\\nFile updated: config/city_profiles.json")

if __name__ == "__main__":
    main()
