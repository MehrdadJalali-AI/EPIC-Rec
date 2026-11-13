# Case studies

Numbers below are copied from `case_1.json`, `case_2.json` and `case_3.json`.
The main run used here is the certified MovieLens-1M study: 2,000 user–target pairs, 200 users, K = 10, alpha = 0.5. It is not the earlier 800-pair prototype.
Entity names are not available: the Satori file in this study is the rehashed RippleNet graph, so a shared person or genre is identified by its relation and rehashed id. The graph stores each link in both directions, so two paths can name the same entity.

## How the pairs were chosen

- success candidates with C optimal, |M| in 2..4, history 15..40, certification confirmed: 213
- of which the target title is on the well-known list: 86
-   candidate user 5489 target 331 'Terminator 2: Judgment Day (1991)' rank 8 |M_C|=2 history 15
-   candidate user 1098 target 497 'Die Hard (1988)' rank 9 |M_C|=2 history 15
-   candidate user 4522 target 530 'E.T. the Extra-Terrestrial (1982)' rank 9 |M_C|=2 history 15
-   candidate user 3806 target 162 'Pulp Fiction (1994)' rank 7 |M_C|=2 history 16
-   candidate user 5596 target 617 'Back to the Future (1985)' rank 7 |M_C|=2 history 16
-   candidate user 3344 target 967 'Saving Private Ryan (1998)' rank 4 |M_C|=2 history 17
-   candidate user 2913 target 268 'Jurassic Park (1993)' rank 9 |M_C|=2 history 17
-   candidate user 3196 target 331 'Terminator 2: Judgment Day (1991)' rank 6 |M_C|=2 history 18
- heuristic screen user 5489 'Terminator 2: Judgment Day (1991)': C1=True C2=True C3=True
- heuristic screen user 1098 'Die Hard (1988)': C1=True C2=False C3=True
- this candidate also has a same-user infeasible target; selecting it
- same user 1098 has 2 other targets with A optimal and B or C proven infeasible
-   infeasible candidate target 331 'Terminator 2: Judgment Day (1991)' rank 1 B=infeasible C=infeasible
-   infeasible candidate target 575 'Star Wars: Episode V - The Empire Strikes Back (1980)' rank 6 B=optimal C=infeasible
- ranking-margin C1-stop: C1 passed=True, C2=False, C3=True
- SELECTED success: user 1098 'Die Hard (1988)' |M_C|=2
- SELECTED infeasible: user 1098 'Terminator 2: Judgment Day (1991)' same_user=True B=infeasible C=infeasible

## Case 1 — success

User 1098 has a training history of 15 movies. K = 10, alpha = 0.5.
The target is Die Hard (1988), originally rank 9.
The original top of the list: 1. Terminator 2: Judgment Day (1991) (6.833281), 2. Die Hard 2 (1990) (6.308484), 3. Hunt for Red October, The (1990) (6.183305), 4. True Lies (1994) (6.144401), 5. Terminator, The (1984) (6.063566).
The strongest paths into the target are Last Action Hero (1993) via film.director.film (rehashed-entity-334, contribution 0.030751), Last Action Hero (1993) via film.film.director (rehashed-entity-334, contribution 0.030751), Whole Nine Yards, The (2000) via film.person_or_entity_appearing_in_film.film (rehashed-entity-1401, contribution 0.025895).
Setting A (C1): status optimal, mask size 2, solve time 0.2813 s, verifier passed True, matches stored status and size True (stored status optimal, stored size 2, stored time 0.6789 s). Masked: Last Action Hero (1993), Big Trouble in Little China (1986). Entering: Enemy of the State (1998).
Setting B (C1+C2): status optimal, mask size 2, solve time 0.1428 s, verifier passed True, matches stored status and size True (stored status optimal, stored size 2, stored time 0.5056 s). Masked: Executive Decision (1996), Dante's Peak (1997). Entering: Raiders of the Lost Ark (1981).
Setting C (C1+C2+C3): status optimal, mask size 2, solve time 0.1331 s, verifier passed True, matches stored status and size True (stored status optimal, stored size 2, stored time 0.5041 s). Masked: River Wild, The (1994), Executive Decision (1996). Entering: Star Wars: Episode VI - Return of the Jedi (1983).
The three history movies that contribute most to the target's score are Matrix, The (1999) (0.744589), Princess Bride, The (1987) (0.710286), Jurassic Park (1993) (0.680122).
After the setting-C mask the top-10 is: Terminator 2: Judgment Day (1991), Terminator, The (1984), Hunt for Red October, The (1990), Star Wars: Episode V - The Empire Strikes Back (1980), Star Wars: Episode IV - A New Hope (1977), Indiana Jones and the Last Crusade (1989), True Lies (1994), Die Hard 2 (1990), Lethal Weapon (1987), Star Wars: Episode VI - Return of the Jedi (1983).
The retained movies that stay are: Terminator 2: Judgment Day (1991), Terminator, The (1984), Hunt for Red October, The (1990), Star Wars: Episode V - The Empire Strikes Back (1980), Star Wars: Episode IV - A New Hope (1977), Indiana Jones and the Last Crusade (1989), True Lies (1994), Die Hard 2 (1990), Lethal Weapon (1987).

## Case 2 — infeasible

User 1098 has a training history of 15 movies. K = 10, alpha = 0.5.
The target is Terminator 2: Judgment Day (1991), originally rank 1.
The original top of the list: 1. Terminator 2: Judgment Day (1991) (6.833281), 2. Die Hard 2 (1990) (6.308484), 3. Hunt for Red October, The (1990) (6.183305), 4. True Lies (1994) (6.144401), 5. Terminator, The (1984) (6.063566).
The strongest paths into the target are Last Action Hero (1993) via film.person_or_entity_appearing_in_film.film (rehashed-entity-506, contribution 0.138105), Last Action Hero (1993) via film.actor.film (rehashed-entity-506, contribution 0.138065), Dante's Peak (1997) via film.person_or_entity_appearing_in_film.film (rehashed-entity-70, contribution 0.103579).
Setting A (C1): status optimal, mask size 5, solve time 0.7348 s, verifier passed True, matches stored status and size True (stored status optimal, stored size 5, stored time 1.1538 s). Masked: Jurassic Park (1993), Last Action Hero (1993), Executive Decision (1996), Dante's Peak (1997), Matrix, The (1999). Entering: There's Something About Mary (1998), Forrest Gump (1994), Enemy of the State (1998).
Setting B (C1+C2): status infeasible, mask size 0, solve time 0.1601 s, verifier passed None, matches stored status and size True (stored status infeasible, stored size None, stored time 0.2483 s). Masked: none. Entering: none.
Setting C (C1+C2+C3): status infeasible, mask size 0, solve time 0.072 s, verifier passed None, matches stored status and size True (stored status infeasible, stored size None, stored time 0.1227 s). Masked: none. Entering: none.
The three history movies that contribute most to the target's score are Matrix, The (1999) (0.970591), Jurassic Park (1993) (0.879734), Last Action Hero (1993) (0.749896).
Infeasibility of setting C: list entanglement: C1 + C2 is already infeasible.
The optimal C1 mask removes the target and also pushes out: Hunt for Red October, The (1990), Star Wars: Episode IV - A New Hope (1977).
Star Wars: Episode IV - A New Hope (1977) also leaves under this particular C1 mask, but allowing that movie to leave does not by itself make C1+C2 feasible.
C1+C2 becomes feasible if any one of these retained movies is allowed to leave: Hunt for Red October, The (1990) (B then optimal, size 6, verifier True).
Hunt for Red October, The (1990) shares no actor, director or writer with the target in this graph. The strongest overlapping link is collaborative co-user from Nutty Professor II: The Klumps (2000) (contribution to this movie 0.011249, to the target 0.001651).

## Case 3 — ranking-margin stopped at C1

Same user and target as case 1: user 1098, Die Hard (1988), originally rank 9.
The ranking-margin heuristic in C1-stop mode masked Executive Decision (1996), Con Air (1997) (2 items) in 0.0002 s.
The verifier on setting A (C1 only) passed: True. On setting C, C1 = True, C2 = False, C3 = True, passed = False.
Retained items that left the top-10: Die Hard 2 (1990).
Retained items that lost every qualifying evidence path: none.
Items that entered the top-10: Raiders of the Lost Ark (1981), Star Wars: Episode VI - Return of the Jedi (1983).
The stored main run recorded ranking-margin only in constrained mode, and C1-stop only for the target-score criterion. This block is a fresh call of greedy(), not a stored record.

