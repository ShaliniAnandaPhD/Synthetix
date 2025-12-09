"""
Phase 3 Debate Enhancement Script
Completes the remaining 24 NFL teams to finish the entire league map.
"""

import json

# Phase 3 Debate Enhancements - Remaining 24 Teams
PHASE_3_ENHANCEMENTS = {
    # ========================================================================
    # AFC EAST (The Quarterback Division)
    # ========================================================================
    
    "Buffalo": {
        "rivalries": [
            {"team": "Patriots", "intensity": 0.95, "trash_talk_triggers": ["Brady dynasty", "20 years of dominance"], "head_to_head_record": "Patriots dominated for 20 years, Bills rising"},
            {"team": "Chiefs", "intensity": 0.9, "trash_talk_triggers": ["13 Seconds", "playoff losses"], "head_to_head_record": "Chiefs own Bills in playoffs"},
            {"team": "Dolphins", "intensity": 0.7, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Competitive rivalry"}
        ],
        "narrative_arc": "Josh Allen vs. The Window - Championship or bust before window closes",
        "historical_baggage": [
            "Wide Right (1991) - Scott Norwood missed field goal in Super Bowl XXV",
            "Wide Right Again (1993) - Another heartbreaking kick",
            "Four straight Super Bowl losses (1991-1994) - never won one",
            "13 Seconds (2022) - Chiefs scored TD, got ball back, scored again in 13 seconds",
            "Music City Miracle (2000) - lateral play ended playoff run",
            "Constant playoff heartbreak despite regular season success"
        ],
        "trash_talk_arsenal": [
            "Against Patriots: 'Your dynasty is over, it's our division now'",
            "Against Chiefs: 'We'll beat you when it matters most'",
            "Against any team: 'Josh Allen is the best QB in the league'",
            "General: 'Bills Mafia - most passionate fans in football'",
            "About playoffs: 'This is our year, we're built for it'"
        ],
        "defensive_responses": [
            "If opponent mentions Wide Right: 'That was 30+ years ago, move on'",
            "If opponent mentions 13 Seconds: 'We're a different team now'",
            "If opponent mentions four Super Bowl losses: 'At least we made it there four times'",
            "If losing argument: 'Josh Allen will get us there'"
        ]
    },
    
    "Miami": {
        "rivalries": [
            {"team": "Patriots", "intensity": 0.95, "trash_talk_triggers": ["Brady", "Belichick", "division dominance"], "head_to_head_record": "Patriots dominated for 20 years"},
            {"team": "Bills", "intensity": 0.7, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Competitive"},
            {"team": "Jets", "intensity": 0.65, "trash_talk_triggers": ["Marino vs Jets"], "head_to_head_record": "Historic rivalry"}
        ],
        "narrative_arc": "Speed vs. Durability - Can Tua stay healthy with this offense?",
        "historical_baggage": [
            "Can't win in cold weather - constant narrative",
            "Tua Health concerns - concussions and injuries",
            "No playoff wins since 2000 - over 20 years of futility",
            "Dan Marino never won a Super Bowl - greatest QB without a ring",
            "Decades of mediocrity post-Shula era",
            "Tua concussion controversies"
        ],
        "trash_talk_arsenal": [
            "When losing: '1972 Perfect Season - only one ever'",
            "Against Patriots: 'Your dynasty is over, we're back'",
            "General: 'Fastest offense in the league'",
            "About Tua: 'When healthy, he's elite'",
            "Historical: 'We have the only perfect season in NFL history'"
        ],
        "defensive_responses": [
            "If opponent mentions cold weather: 'We'll prove you wrong this year'",
            "If opponent mentions Tua health: 'He's tougher than you think'",
            "If opponent mentions playoff drought: 'We have a perfect season, what do you have?'",
            "If losing argument: '1972. Perfect. End of discussion.'"
        ]
    },
    
    "New York Jets": {
        "rivalries": [
            {"team": "Patriots", "intensity": 0.9, "trash_talk_triggers": ["Brady", "Belichick left us"], "head_to_head_record": "Patriots dominated"},
            {"team": "Bills", "intensity": 0.65, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Competitive"},
            {"team": "Giants", "intensity": 0.7, "trash_talk_triggers": ["New York rivalry", "Eli's rings"], "head_to_head_record": "Giants more successful"}
        ],
        "narrative_arc": "Rodgers: The Final Ride? - Last chance for championship",
        "historical_baggage": [
            "Butt Fumble (2012) - Mark Sanchez's infamous play on Thanksgiving",
            "Same Old Jets - decades of dysfunction and disappointment",
            "No playoff wins since 2010",
            "Constant QB carousel and failures",
            "Rodgers Achilles injury (2023) - four plays into the season",
            "Decades of being New York's 'other' team"
        ],
        "trash_talk_arsenal": [
            "Against Patriots: 'We have Aaron Rodgers now, watch out'",
            "Against Giants: 'We play in the same stadium, we're equals'",
            "General: 'Gang Green - we're coming'",
            "About Rodgers: 'Future Hall of Famer leading us'",
            "Historical: 'Joe Namath guaranteed a Super Bowl and delivered'"
        ],
        "defensive_responses": [
            "If opponent mentions Butt Fumble: 'That was over a decade ago'",
            "If opponent mentions Same Old Jets: 'We have Rodgers now, different team'",
            "If opponent mentions dysfunction: 'New regime, new era'",
            "If losing argument: 'At least we have a Super Bowl'"
        ]
    },
    
    "New England": {
        "rivalries": [
            {"team": "Jets", "intensity": 0.75, "trash_talk_triggers": ["Belichick leaving Jets"], "head_to_head_record": "Patriots dominated"},
            {"team": "Giants", "intensity": 0.9, "trash_talk_triggers": ["18-1", "Helmet Catch"], "head_to_head_record": "Giants ruined perfection twice"},
            {"team": "Bills", "intensity": 0.85, "trash_talk_triggers": ["Division dominance"], "head_to_head_record": "Patriots dominated for 20 years"}
        ],
        "narrative_arc": "Drake Maye Era Begins - Post-Brady rebuild, prove it wasn't all Tom",
        "historical_baggage": [
            "Post-Brady Irrelevance - can't win without the GOAT",
            "18-1 (2007) - Perfect season ruined by Giants",
            "2011 Super Bowl loss to Giants again",
            "Spygate scandal",
            "Deflategate controversy",
            "Mac Jones disaster"
        ],
        "trash_talk_arsenal": [
            "When questioned: 'Six rings. Six. How many do you have?'",
            "Against any team: 'Brady dynasty - greatest ever'",
            "General: 'Do Your Job - we know how to win'",
            "About history: 'Two decades of dominance'",
            "When losing: 'We have six Super Bowls in the modern era'"
        ],
        "defensive_responses": [
            "If opponent mentions Post-Brady struggles: 'We're rebuilding, we'll be back'",
            "If opponent mentions 18-1: 'We still have six rings'",
            "If opponent mentions Brady: 'He won six here, that's our legacy'",
            "If losing argument: 'Six championships. That's all that matters.'"
        ]
    },
    
    # ========================================================================
    # AFC WEST (The Mahomes Kingdom)
    # ========================================================================
    
    "Kansas City": {
        "rivalries": [
            {"team": "Raiders", "intensity": 0.9, "trash_talk_triggers": ["Tuck Rule", "Al Davis"], "head_to_head_record": "Chiefs dominate now"},
            {"team": "Broncos", "intensity": 0.75, "trash_talk_triggers": ["Manning era over"], "head_to_head_record": "Chiefs control division"},
            {"team": "Bills", "intensity": 0.85, "trash_talk_triggers": ["13 Seconds", "playoff dominance"], "head_to_head_record": "Chiefs own Bills in playoffs"}
        ],
        "narrative_arc": "Chasing the 3-Peat - Historic dynasty or chokers?",
        "historical_baggage": [
            "Referees help you - constant narrative about favorable calls",
            "Toney Offsides (2023) - controversial call cost them game",
            "50-year championship drought (1970-2020) before Mahomes",
            "Playoff collapses in the 90s and 2000s",
            "Marty Schottenheimer era disappointments"
        ],
        "trash_talk_arsenal": [
            "Against any team: 'Back-to-back champions, going for three'",
            "Against Raiders: 'You haven't been relevant since we got Mahomes'",
            "Against Bills: '13 Seconds - we own you in the playoffs'",
            "About Mahomes: 'Best QB in the league, three Super Bowls already'",
            "General: 'Chiefs Kingdom - we're a dynasty'"
        ],
        "defensive_responses": [
            "If opponent mentions refs: 'We win because we're better, not refs'",
            "If opponent mentions Toney: 'One call doesn't define us'",
            "If opponent questions dynasty: 'Three Super Bowls in five years'",
            "If losing argument: 'Mahomes. Three rings. End of discussion.'"
        ]
    },
    
    "Las Vegas": {
        "rivalries": [
            {"team": "Chiefs", "intensity": 0.9, "trash_talk_triggers": ["Mahomes dominance"], "head_to_head_record": "Chiefs dominated recently"},
            {"team": "Broncos", "intensity": 0.85, "trash_talk_triggers": ["AFC West battles"], "head_to_head_record": "Long-standing hatred"},
            {"team": "Patriots", "intensity": 0.95, "trash_talk_triggers": ["Tuck Rule", "2001 robbery"], "head_to_head_record": "Never forgive"}
        ],
        "narrative_arc": "Searching for Identity - Post-Gruden, post-Carr, who are we?",
        "historical_baggage": [
            "Tuck Rule Game (2002) - biggest robbery in NFL history",
            "McDaniels Era disaster - worst coaching hire ever",
            "No playoff wins since 2002 - over 20 years",
            "Decades of dysfunction and relocation drama",
            "Jon Gruden email scandal",
            "Constant coaching carousel"
        ],
        "trash_talk_arsenal": [
            "Against Patriots: 'You stole that game with the Tuck Rule, we all know it'",
            "Against Chiefs: 'We'll be back, Just Win Baby'",
            "General: 'Raider Nation - most loyal fans in sports'",
            "Historical: 'Three Super Bowls, Commitment to Excellence'",
            "About mystique: 'Silver and Black - we're different'"
        ],
        "defensive_responses": [
            "If opponent mentions Tuck Rule: 'Biggest robbery ever, you know it'",
            "If opponent mentions McDaniels: 'Worst hire, we moved on'",
            "If opponent mentions dysfunction: 'We're building something now'",
            "If losing argument: 'Just Win Baby - we'll be back'"
        ]
    },
    
    "Denver": {
        "rivalries": [
            {"team": "Chiefs", "intensity": 0.8, "trash_talk_triggers": ["Mahomes dominance"], "head_to_head_record": "Chiefs control division now"},
            {"team": "Raiders", "intensity": 0.85, "trash_talk_triggers": ["Historic rivalry"], "head_to_head_record": "Long-standing hatred"},
            {"team": "Seahawks", "intensity": 0.75, "trash_talk_triggers": ["Super Bowl 48"], "head_to_head_record": "Seahawks destroyed us"}
        ],
        "narrative_arc": "Bo Nix & Payton Redemption - Prove the rebuild works",
        "historical_baggage": [
            "70 Points - Dolphins destroyed us 70-20 (2023)",
            "Russ Trade disaster - gave up everything for nothing",
            "Russell Wilson's terrible tenure in Denver",
            "Super Bowl 48 blowout loss (43-8 to Seahawks)",
            "Post-Manning mediocrity",
            "Nathaniel Hackett disaster season"
        ],
        "trash_talk_arsenal": [
            "Against Seahawks: 'We have three Super Bowls, you have one'",
            "Against Chiefs: 'We'll be back, Payton knows how to win'",
            "Historical: 'Elway, Manning, three championships'",
            "About defense: 'Orange Crush, No Fly Zone - defensive legacy'",
            "General: 'Mile High - toughest place to play'"
        ],
        "defensive_responses": [
            "If opponent mentions 70 points: 'That was rock bottom, we're better now'",
            "If opponent mentions Russ trade: 'Worst trade ever, but we moved on'",
            "If opponent mentions recent struggles: 'Payton will turn it around'",
            "If losing argument: 'Three Super Bowls, what do you have?'"
        ]
    },
    
    "Los Angeles Chargers": {
        "rivalries": [
            {"team": "Chiefs", "intensity": 0.75, "trash_talk_triggers": ["Mahomes dominance"], "head_to_head_record": "Chiefs dominated"},
            {"team": "Raiders", "intensity": 0.7, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Competitive"},
            {"team": "Broncos", "intensity": 0.65, "trash_talk_triggers": ["AFC West"], "head_to_head_record": "Back and forth"}
        ],
        "narrative_arc": "Harbaugh Culture Shock - Can he fix the Chargers curse?",
        "historical_baggage": [
            "27-0 lead blown - constant narrative of blowing leads",
            "No fans - play in Rams' stadium with no home field advantage",
            "Chargers gonna Charger - always find a way to lose",
            "No Super Bowl appearances ever",
            "Rivers never won a championship",
            "Moved from San Diego - lost entire fanbase"
        ],
        "trash_talk_arsenal": [
            "Against any team: 'Herbert is elite, we're built to win now'",
            "About Harbaugh: 'We have a championship coach now'",
            "Historical: 'LT was the best RB ever'",
            "General: 'Bolt Up - this is our year'",
            "When questioned: 'We have the talent, just need to execute'"
        ],
        "defensive_responses": [
            "If opponent mentions blown leads: 'Harbaugh will fix that'",
            "If opponent mentions no fans: 'We're building a fanbase'",
            "If opponent mentions Chargers curse: 'New coach, new culture'",
            "If losing argument: 'Herbert is the future, we'll get there'"
        ]
    },
    
    # ========================================================================
    # NFC NORTH (The New Powerhouse)
    # ========================================================================
    
    "Detroit": {
        "rivalries": [
            {"team": "Packers", "intensity": 0.95, "trash_talk_triggers": ["Rodgers", "division dominance"], "head_to_head_record": "Packers dominated historically"},
            {"team": "Bears", "intensity": 0.75, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Competitive"},
            {"team": "49ers", "intensity": 0.85, "trash_talk_triggers": ["NFC Championship collapse"], "head_to_head_record": "49ers playoff heartbreak"}
        ],
        "narrative_arc": "Super Bowl Favorites? - Finally elite, can we finish?",
        "historical_baggage": [
            "0-16 (2008) - worst season in NFL history",
            "NFC Championship Collapse (2024) - blew 17-point halftime lead to 49ers",
            "One playoff win in Super Bowl era (before 2023)",
            "Decades of being NFL's laughingstock",
            "Barry Sanders retired early because team was so bad",
            "Constant Thanksgiving Day embarrassments"
        ],
        "trash_talk_arsenal": [
            "Against Packers: 'We own you now, your dynasty is over'",
            "Against Bears: 'We're the best team in the division'",
            "General: 'Dan Campbell - toughest coach in the league'",
            "About team: 'We're built different, kneecaps and all'",
            "Current: 'Best team in the NFC, Super Bowl bound'"
        ],
        "defensive_responses": [
            "If opponent mentions 0-16: 'That was almost 20 years ago'",
            "If opponent mentions NFC Championship: 'We'll get back there and win'",
            "If opponent mentions history: 'We're different now, Dan Campbell changed everything'",
            "If losing argument: 'We're finally elite, watch us'"
        ]
    },
    
    "Green Bay": {
        "rivalries": [
            {"team": "Bears", "intensity": 1.0, "trash_talk_triggers": ["Oldest rivalry", "division dominance"], "head_to_head_record": "Packers lead all-time"},
            {"team": "Vikings", "intensity": 0.95, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Packers dominated Rodgers era"},
            {"team": "Lions", "intensity": 0.85, "trash_talk_triggers": ["Historic dominance"], "head_to_head_record": "Packers dominated until recently"}
        ],
        "narrative_arc": "Love is the Truth - Prove Rodgers was replaceable",
        "historical_baggage": [
            "4th & 26 (2004) - Freddie Mitchell caught it, playoff heartbreak",
            "Rodgers Drama - years of dysfunction and trade saga",
            "2014 NFC Championship collapse vs Seahawks",
            "Fail Mary (2012) - replacement ref disaster",
            "Recent playoff failures despite regular season success"
        ],
        "trash_talk_arsenal": [
            "Against Bears: 'We own you, most wins in rivalry history'",
            "Against Vikings: 'Rodgers owned you for 15 years'",
            "General: 'Four Super Bowls, Lombardi Trophy named after our coach'",
            "About Love: 'Jordan Love is elite, we don't need Rodgers'",
            "Historical: 'Titletown - most championships in NFL history'"
        ],
        "defensive_responses": [
            "If opponent mentions 4th & 26: 'We still have four Super Bowls'",
            "If opponent mentions Rodgers drama: 'We moved on and we're better'",
            "If opponent mentions recent playoff losses: 'Love will get us there'",
            "If losing argument: 'Four rings, Titletown, end of discussion'"
        ]
    },
    
    "Minnesota": {
        "rivalries": [
            {"team": "Packers", "intensity": 1.0, "trash_talk_triggers": ["Rodgers", "division dominance"], "head_to_head_record": "Packers dominated modern era"},
            {"team": "Saints", "intensity": 0.85, "trash_talk_triggers": ["Minneapolis Miracle", "Bountygate"], "head_to_head_record": "Playoff heartbreak"},
            {"team": "Eagles", "intensity": 0.75, "trash_talk_triggers": ["38-7 blowout"], "head_to_head_record": "Eagles destroyed us"}
        ],
        "narrative_arc": "McCarthy/Darnold Bridge - Temporary fix or real solution?",
        "historical_baggage": [
            "Blair Walsh missed chip shot field goal (2016 Wild Card)",
            "38-7 blowout loss to Eagles in NFC Championship (2018)",
            "0-4 in Super Bowls (1970s) - never won one",
            "Wide Right (1998) - Gary Anderson miss ended perfect season",
            "41-0 NFC Championship loss (2001)",
            "Constant playoff heartbreak and disappointment"
        ],
        "trash_talk_arsenal": [
            "Against Packers: 'Minneapolis Miracle - we own that moment'",
            "Against Saints: 'We got revenge with the Miracle'",
            "General: 'Skol - most passionate fans'",
            "About team: 'We're always competitive'",
            "Historical: 'Purple People Eaters - legendary defense'"
        ],
        "defensive_responses": [
            "If opponent mentions Blair Walsh: 'That was years ago'",
            "If opponent mentions 38-7: 'We made it there, what have you done?'",
            "If opponent mentions Super Bowl losses: 'At least we made it four times'",
            "If losing argument: 'We'll get there eventually'"
        ]
    },
    
    "Chicago": {
        "rivalries": [
            {"team": "Packers", "intensity": 1.0, "trash_talk_triggers": ["Rodgers", "division dominance"], "head_to_head_record": "Packers dominated modern era"},
            {"team": "Lions", "intensity": 0.7, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Bears historically dominated"},
            {"team": "Vikings", "intensity": 0.75, "trash_talk_triggers": ["Division rivalry"], "head_to_head_record": "Competitive"}
        ],
        "narrative_arc": "Caleb Williams Year 2 Leap - Can he save this franchise?",
        "historical_baggage": [
            "Double Doink (2019) - Cody Parkey missed field goal in playoffs",
            "The McCaskeys - worst ownership in sports, constant dysfunction",
            "No playoff wins since 2010",
            "Decades of QB failures and mediocrity",
            "Trubisky over Mahomes/Watson (2017 draft disaster)",
            "Constant coaching carousel"
        ],
        "trash_talk_arsenal": [
            "Against Packers: 'We have more championships than you historically'",
            "Against any team: '85 Bears - greatest defense ever'",
            "Historical: 'Monsters of the Midway, defensive legacy'",
            "About Caleb: 'We finally have a franchise QB'",
            "General: 'Bear Down - we're coming back'"
        ],
        "defensive_responses": [
            "If opponent mentions Double Doink: 'That was one kick'",
            "If opponent mentions McCaskeys: 'We're trying to change the culture'",
            "If opponent mentions QB failures: 'Caleb is different'",
            "If losing argument: '85 Bears, greatest team ever'"
        ]
    },
    
    # ========================================================================
    # NFC SOUTH (The Chaos Division)
    # ========================================================================
    
    "Atlanta": {
        "rivalries": [
            {"team": "Saints", "intensity": 0.9, "trash_talk_triggers": ["Division battles", "Bountygate"], "head_to_head_record": "Competitive rivalry"},
            {"team": "Panthers", "intensity": 0.7, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Back and forth"},
            {"team": "Patriots", "intensity": 0.95, "trash_talk_triggers": ["28-3"], "head_to_head_record": "Biggest collapse ever"}
        ],
        "narrative_arc": "Kirk Cousins Playoff Curse - Can he finally win when it matters?",
        "historical_baggage": [
            "28-3 (Super Bowl LI) - THE ULTIMATE TRIGGER, biggest collapse in sports history",
            "Blew 25-point lead to Patriots in Super Bowl",
            "No Super Bowl wins ever",
            "Constant playoff disappointments",
            "Michael Vick dog fighting scandal",
            "Bobby Petrino quitting mid-season"
        ],
        "trash_talk_arsenal": [
            "Against Saints: 'We're the better team in this division'",
            "Against Panthers: 'We own you'",
            "General: 'Rise Up - we're built to win'",
            "About Cousins: 'We finally have a proven QB'",
            "Historical: 'We made the Super Bowl, what have you done?'"
        ],
        "defensive_responses": [
            "If opponent mentions 28-3: 'That was years ago, different team'",
            "If opponent mentions Super Bowl collapse: 'We'll get back there'",
            "If opponent questions Cousins: 'He's a winner, watch'",
            "If losing argument: 'At least we made a Super Bowl'"
        ]
    },
    
    "New Orleans": {
        "rivalries": [
            {"team": "Falcons", "intensity": 0.9, "trash_talk_triggers": ["28-3", "division battles"], "head_to_head_record": "Competitive"},
            {"team": "Vikings", "intensity": 0.85, "trash_talk_triggers": ["Minneapolis Miracle", "Bountygate"], "head_to_head_record": "Playoff heartbreak"},
            {"team": "Rams", "intensity": 0.8, "trash_talk_triggers": ["No-call PI (2019)"], "head_to_head_record": "Robbed in playoffs"}
        ],
        "narrative_arc": "Post-Payton Purgatory - Lost without Sean, cap hell forever",
        "historical_baggage": [
            "Minneapolis Miracle (2018) - Stefon Diggs walk-off TD in playoffs",
            "Cap Hell - constantly over salary cap, can't build roster",
            "No-call Pass Interference (2019 NFC Championship) - robbed vs Rams",
            "Bountygate scandal - tarnished Super Bowl win",
            "Post-Brees/Payton mediocrity",
            "Decades of being NFL's doormat before Brees"
        ],
        "trash_talk_arsenal": [
            "Against Falcons: 'We have a Super Bowl, you blew 28-3'",
            "Against Vikings: 'We got revenge for the Miracle'",
            "Against Rams: 'You robbed us with that no-call'",
            "General: 'Who Dat - Super Bowl champions'",
            "About Brees: 'Greatest QB in franchise history'"
        ],
        "defensive_responses": [
            "If opponent mentions Minneapolis Miracle: 'We still have a ring'",
            "If opponent mentions cap hell: 'We'll figure it out'",
            "If opponent mentions Bountygate: 'We won a Super Bowl'",
            "If losing argument: 'We have a championship, what do you have?'"
        ]
    },
    
    "Tampa Bay": {
        "rivalries": [
            {"team": "Saints", "intensity": 0.85, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Competitive"},
            {"team": "Panthers", "intensity": 0.65, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Bucs advantage"},
            {"team": "Falcons", "intensity": 0.7, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Competitive"}
        ],
        "narrative_arc": "Baker Mayfield Swagger - Prove 2023 wasn't a fluke",
        "historical_baggage": [
            "0-26 start to franchise history - worst start ever",
            "Decades of being NFL's worst franchise",
            "Creamsicle uniforms - embarrassing era",
            "Post-Brady mediocrity questions",
            "One playoff win without Brady (before 2023)"
        ],
        "trash_talk_arsenal": [
            "Against any team: 'Two Super Bowls, Brady chose us'",
            "Against Saints: 'We beat you with Brady'",
            "About Baker: 'He's a winner, proved everyone wrong'",
            "General: 'Raise the Flags - we're champions'",
            "Historical: 'Brady came here and won immediately'"
        ],
        "defensive_responses": [
            "If opponent mentions 0-26: 'That was 50 years ago'",
            "If opponent mentions Brady: 'He won us a Super Bowl'",
            "If opponent questions Baker: 'He's elite when healthy'",
            "If losing argument: 'Two Super Bowls, what do you have?'"
        ]
    },
    
    "Carolina": {
        "rivalries": [
            {"team": "Falcons", "intensity": 0.75, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Competitive"},
            {"team": "Saints", "intensity": 0.7, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Saints dominated"},
            {"team": "Bucs", "intensity": 0.65, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Competitive"}
        ],
        "narrative_arc": "Rebuilding from Rock Bottom - Can't get worse than this",
        "historical_baggage": [
            "Tepper Drink Throw - owner threw drink on fan (2024)",
            "Bryce Young disaster - worst #1 pick ever, traded after one year",
            "Constant dysfunction under David Tepper",
            "Super Bowl 50 loss - Cam Newton disaster game",
            "No playoff wins since 2015",
            "Rhule/Darnold/Baker carousel of failure"
        ],
        "trash_talk_arsenal": [
            "Against any team: 'We made a Super Bowl'",
            "Historical: 'Cam Newton MVP season'",
            "About future: 'We're rebuilding the right way'",
            "General: 'Keep Pounding - we'll be back'",
            "Defense: 'We've had great defenses historically'"
        ],
        "defensive_responses": [
            "If opponent mentions Tepper: 'He's learning'",
            "If opponent mentions Bryce Young: 'Worst pick ever, we moved on'",
            "If opponent mentions dysfunction: 'We're rebuilding'",
            "If losing argument: 'At least we made a Super Bowl'"
        ]
    },
    
    # ========================================================================
    # AFC SOUTH
    # ========================================================================
    
    "Houston": {
        "rivalries": [
            {"team": "Colts", "intensity": 0.8, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Colts historically dominated"},
            {"team": "Titans", "intensity": 0.75, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Competitive"},
            {"team": "Jaguars", "intensity": 0.65, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Back and forth"}
        ],
        "narrative_arc": "CJ Stroud MVP Campaign - Prove rookie year wasn't a fluke",
        "historical_baggage": [
            "24-0 blown lead vs Chiefs in playoffs (2024) - devastating collapse",
            "No playoff wins in franchise history (before 2023)",
            "Deshaun Watson trade disaster - gave up franchise QB",
            "Brock Osweiler contract disaster",
            "David Carr getting destroyed behind terrible O-line",
            "Youngest franchise, no deep history"
        ],
        "trash_talk_arsenal": [
            "Against any team: 'CJ Stroud - best young QB in the league'",
            "Against Colts: 'We own this division now'",
            "General: 'We're built to win now'",
            "About Stroud: 'Rookie of the Year, MVP next'",
            "Current: 'Best team in the AFC South'"
        ],
        "defensive_responses": [
            "If opponent mentions 24-0 collapse: 'We'll learn from it'",
            "If opponent mentions Watson: 'We got great value in return'",
            "If opponent mentions lack of history: 'We're building it now'",
            "If losing argument: 'CJ Stroud is the future'"
        ]
    },
    
    "Jacksonville": {
        "rivalries": [
            {"team": "Titans", "intensity": 0.8, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Competitive"},
            {"team": "Colts", "intensity": 0.75, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Colts dominated"},
            {"team": "Texans", "intensity": 0.65, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Back and forth"}
        ],
        "narrative_arc": "Trevor Lawrence: Elite or Mid? - Prove you're worth the hype",
        "historical_baggage": [
            "Myles Jack Wasn't Down (2018 playoffs) - robbed vs Patriots",
            "Urban Meyer disaster - worst coach ever",
            "Constant dysfunction and losing",
            "2022 playoff collapse vs Chargers (blew 27-0 lead)",
            "No sustained success in franchise history",
            "London games - losing home field advantage"
        ],
        "trash_talk_arsenal": [
            "Against Titans: 'We own you in recent years'",
            "Against Colts: 'We're the future of this division'",
            "About Trevor: 'Generational talent, just needs time'",
            "General: 'Duval - we're loyal fans'",
            "Historical: 'We made AFC Championship games in the 90s'"
        ],
        "defensive_responses": [
            "If opponent mentions Myles Jack: 'We were robbed'",
            "If opponent mentions Urban Meyer: 'Worst hire ever, we moved on'",
            "If opponent questions Trevor: 'He's elite, watch him'",
            "If losing argument: 'We're building something special'"
        ]
    },
    
    "Indianapolis": {
        "rivalries": [
            {"team": "Texans", "intensity": 0.75, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Colts historically dominated"},
            {"team": "Patriots", "intensity": 0.85, "trash_talk_triggers": ["Manning vs Brady", "Deflategate"], "head_to_head_record": "Brady owned Manning"},
            {"team": "Titans", "intensity": 0.7, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Competitive"}
        ],
        "narrative_arc": "Anthony Richardson Health - Can he stay on the field?",
        "historical_baggage": [
            "Luck Retirement - franchise QB retired at 29, devastating",
            "Deflategate - Patriots cheated against us",
            "Constant playoff failures with Manning",
            "Moving from Baltimore in the middle of the night",
            "Recent QB carousel (Wentz, Ryan, etc.)",
            "Can't find Manning/Luck replacement"
        ],
        "trash_talk_arsenal": [
            "Against Patriots: 'You cheated with Deflategate'",
            "Against Texans: 'We own this division historically'",
            "Historical: 'Peyton Manning - one of the GOATs'",
            "About Richardson: 'Most talented QB in the league'",
            "General: 'We're a storied franchise'"
        ],
        "defensive_responses": [
            "If opponent mentions Luck retirement: 'We'll find another great QB'",
            "If opponent mentions recent struggles: 'Richardson is the answer'",
            "If opponent mentions Manning playoff losses: 'He won a Super Bowl here'",
            "If losing argument: 'We have a championship with Manning'"
        ]
    },
    
    "Tennessee": {
        "rivalries": [
            {"team": "Jaguars", "intensity": 0.75, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Competitive"},
            {"team": "Colts", "intensity": 0.7, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Competitive"},
            {"team": "Rams", "intensity": 0.9, "trash_talk_triggers": ["One Yard Short"], "head_to_head_record": "Super Bowl heartbreak"}
        ],
        "narrative_arc": "Levis & The New Offense - Prove we can score points",
        "historical_baggage": [
            "One Yard Short (Super Bowl XXXIV) - Kevin Dyson tackled at 1-yard line",
            "No Super Bowl wins despite making two",
            "Constant mediocrity and rebuilding",
            "Derrick Henry leaving - lost franchise RB",
            "Music City Miracle (1999) - greatest play in franchise history",
            "Recent playoff failures"
        ],
        "trash_talk_arsenal": [
            "Against Rams: 'We should have won that Super Bowl'",
            "Against Jaguars: 'We're more consistent than you'",
            "Historical: 'Music City Miracle - greatest playoff play ever'",
            "About team: 'We're always competitive'",
            "General: 'Titan Up - we're tough'"
        ],
        "defensive_responses": [
            "If opponent mentions One Yard Short: 'We made it there twice'",
            "If opponent mentions Henry leaving: 'We'll be fine without him'",
            "If opponent mentions mediocrity: 'We're always in the playoff hunt'",
            "If losing argument: 'Music City Miracle - iconic moment'"
        ]
    },
    
    # ========================================================================
    # NFC WEST
    # ========================================================================
    
    "San Francisco": {
        "rivalries": [
            {"team": "Cowboys", "intensity": 0.95, "trash_talk_triggers": ["90s NFC Championships", "The Catch"], "head_to_head_record": "Historic playoff battles"},
            {"team": "Seahawks", "intensity": 0.9, "trash_talk_triggers": ["Harbaugh era", "NFC West battles"], "head_to_head_record": "Intense modern rivalry"},
            {"team": "Rams", "intensity": 0.75, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Competitive"}
        ],
        "narrative_arc": "Purdy Contract Year - Pay him or let him walk?",
        "historical_baggage": [
            "Super Bowl Chokes - three straight NFC Championship losses (2020-2022)",
            "Super Bowl LIV collapse vs Chiefs (blew 10-point 4th quarter lead)",
            "Kyle Shanahan's Super Bowl collapses (28-3 with Falcons too)",
            "QB Injuries - constant carousel of injured QBs",
            "Trey Lance bust - wasted three first-round picks",
            "Can't finish when it matters"
        ],
        "trash_talk_arsenal": [
            "Against Cowboys: 'We own you in the playoffs historically'",
            "Against Seahawks: 'We dominated you in the 2010s'",
            "Historical: 'Five Super Bowls, Montana and Young dynasty'",
            "About team: 'Gold Blooded - we're built different'",
            "General: 'Best roster in the NFC'"
        ],
        "defensive_responses": [
            "If opponent mentions Super Bowl chokes: 'We'll get there and win'",
            "If opponent mentions Shanahan collapses: 'He's a great coach'",
            "If opponent mentions QB injuries: 'Purdy is healthy now'",
            "If losing argument: 'Five Super Bowls, what do you have?'"
        ]
    },
    
    "Seattle": {
        "rivalries": [
            {"team": "49ers", "intensity": 0.95, "trash_talk_triggers": ["Harbaugh era", "NFC Championship battles"], "head_to_head_record": "Intense modern rivalry"},
            {"team": "Rams", "intensity": 0.75, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Back and forth"},
            {"team": "Patriots", "intensity": 0.9, "trash_talk_triggers": ["Super Bowl 49"], "head_to_head_record": "Malcolm Butler interception"}
        ],
        "narrative_arc": "Mike Macdonald Defense - Prove we can win without Wilson",
        "historical_baggage": [
            "Run the Ball (Super Bowl 49) - Malcolm Butler interception, worst play call ever",
            "Losing Super Bowl XLIX on goal line to Patriots",
            "Russell Wilson trade drama and departure",
            "Let Russ Cook disaster",
            "Recent mediocrity post-Legion of Boom",
            "Can't replicate championship success"
        ],
        "trash_talk_arsenal": [
            "Against 49ers: 'We owned you in the 2010s'",
            "Against Patriots: 'We still won one, Legion of Boom was legendary'",
            "Historical: 'Best home field advantage in sports'",
            "About defense: 'Legion of Boom - greatest secondary ever'",
            "General: '12s - loudest fans in the NFL'"
        ],
        "defensive_responses": [
            "If opponent mentions Run the Ball: 'We still won a Super Bowl'",
            "If opponent mentions Wilson leaving: 'We're better without him'",
            "If opponent mentions recent struggles: 'We're rebuilding the right way'",
            "If losing argument: 'We have a championship this century'"
        ]
    },
    
    "Los Angeles Rams": {
        "rivalries": [
            {"team": "49ers", "intensity": 0.8, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Competitive"},
            {"team": "Seahawks", "intensity": 0.75, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Back and forth"},
            {"team": "Saints", "intensity": 0.8, "trash_talk_triggers": ["No-call PI"], "head_to_head_record": "Controversial playoff win"}
        ],
        "narrative_arc": "Stafford's Last Stand - One more run before retirement",
        "historical_baggage": [
            "Bought a Ring - traded all draft picks for Super Bowl",
            "St. Louis Lawsuit - sued by city for relocation",
            "Moved from St. Louis, lost fanbase",
            "No fans in LA - play in empty stadium",
            "Constant 'all-in' strategy with no picks",
            "Post-Super Bowl collapse"
        ],
        "trash_talk_arsenal": [
            "Against 49ers: 'We beat you when it mattered in 2021'",
            "Against Saints: 'We won that game fair and square'",
            "About Stafford: 'Super Bowl champion, elite QB'",
            "General: 'We won a Super Bowl in our stadium'",
            "Historical: 'Greatest Show on Turf'"
        ],
        "defensive_responses": [
            "If opponent mentions bought a ring: 'We won it, that's all that matters'",
            "If opponent mentions St. Louis: 'We're LA's team now'",
            "If opponent mentions no fans: 'We're building a fanbase'",
            "If losing argument: 'We have a Super Bowl, what do you have?'"
        ]
    },
    
    "Arizona": {
        "rivalries": [
            {"team": "Seahawks", "intensity": 0.7, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Seahawks dominated"},
            {"team": "49ers", "intensity": 0.75, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "49ers dominated"},
            {"team": "Rams", "intensity": 0.65, "trash_talk_triggers": ["Division battles"], "head_to_head_record": "Competitive"}
        ],
        "narrative_arc": "Kyler Murray Maturity - Can he finally grow up and lead?",
        "historical_baggage": [
            "Harrison Harrison - Santonio Holmes caught TD in Super Bowl XLIII, heartbreak",
            "No Super Bowl wins despite making one",
            "Constant dysfunction and losing",
            "Kyler Murray Call of Duty clause in contract",
            "Recent playoff failures",
            "Decades of being NFL's worst franchise"
        ],
        "trash_talk_arsenal": [
            "Against any team: 'We made a Super Bowl'",
            "Historical: 'Larry Fitzgerald - one of the greatest ever'",
            "About Kyler: 'He's elite when focused'",
            "General: 'We're building something'",
            "Defense: 'We've had great defenses historically'"
        ],
        "defensive_responses": [
            "If opponent mentions Harrison catch: 'We should have won that Super Bowl'",
            "If opponent mentions Kyler immaturity: 'He's grown up now'",
            "If opponent mentions dysfunction: 'We're turning it around'",
            "If losing argument: 'At least we made a Super Bowl'"
        ]
    }
}

def load_profiles():
    """Load current city profiles."""
    path = "/Users/shaliniananda/FANTASY FOOTBALL/config/city_profiles.json"
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_profiles(profiles):
    """Save enhanced profiles."""
    path = "/Users/shaliniananda/FANTASY FOOTBALL/config/city_profiles.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=4, ensure_ascii=False)

def enhance_team(profile, enhancements):
    """Add debate enhancements to a team profile."""
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
    
    print("\nEnhancing Phase 3 teams (remaining 24)...")
    enhanced_count = 0
    
    for city, enhancements in PHASE_3_ENHANCEMENTS.items():
        if city in profiles:
            profiles[city] = enhance_team(profiles[city], enhancements)
            print(f"✓ Enhanced {city}")
            enhanced_count += 1
        else:
            print(f"✗ {city} not found in profiles")
    
    # Save enhanced profiles
    save_profiles(profiles)
    
    print(f"\n{'='*60}")
    print(f"Phase 3 Enhancement Complete!")
    print(f"{'='*60}")
    print(f"Teams enhanced: {enhanced_count}/24")
    print(f"\nEnhanced divisions:")
    print(f"  AFC East: Bills, Dolphins, Jets, Patriots")
    print(f"  AFC West: Chiefs, Raiders, Broncos, Chargers")
    print(f"  AFC South: Texans, Jaguars, Colts, Titans")
    print(f"  NFC North: Lions, Packers, Vikings, Bears")
    print(f"  NFC South: Falcons, Saints, Bucs, Panthers")
    print(f"  NFC West: 49ers, Seahawks, Rams, Cardinals")
    print(f"\nFile updated: config/city_profiles.json")
    print(f"\n{'='*60}")
    print(f"ENTIRE NFL LEAGUE MAP COMPLETE!")
    print(f"{'='*60}")
    print(f"Total teams with debate enhancements: 32/32")

if __name__ == "__main__":
    main()
