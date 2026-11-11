# Task: implement honest, liar-aware, distribution-aware, and observe-then-choose player

Implement a strategy to select people to propose based on the strategy below:

### What to keep track of

- For each person we keep an estimated value that is taken from the formula, every projection they have shown, and every stated ideal they have given
- Distributions for each dimension
- Threshold based on the distribution of estimates for all people. The value goes down as rounds go by and single people become less

### Distributions

- Build distributions from projections and ads. The number of rounds can be too few to generate unbiased distributions, so start from normal distributions and for each round adjust them by using all players' advertisements and the projection given in the dates
- For each dimension's distribution assume there are liars: monotonic dimensions are more likely to have liars showing bigger numbers. Non-monotinic dimensions are likely honest as showing off fake values would affect both parties

### Honesty

- The player should prioritize honesty every round. At the start it prefers to show its highest/lowest value in a non-monotic dimension (honesty about these dimensions favor both parties) and sets its ideal as the highest/lowest value for a non-monotonic dimension (same idea)

### Decisions

- For each date if the estimated score of a person is negative we mark them as incompatible.
- If a person has a positive score but has lied (see liar detection), we return an empty decision and dont add them to the choose_dates list for the other round
- If the score is positive and is the highest we've seen we then mark them as again. If its positive but is not the highest we just return the empty decision but keep them in the choose_dates lisg for the next round.
- We just propose in these cases: if we are in the observe phase then only propose if the other person proposes and the estimate is nonnegative, if we are in the decide phase propose to people that meet/exceed the threshold.
-

### Observe-then-choose

- Based on the secretary problem, the player uses its first rounds to observe and the last ones to choose who to marry.
- There may me few rounds to observing without deciding anything is not ideal. Instead of using the 37% of round to just observe, we use the floor(sqrt(P)) where P is the number of unique players dated. When the player reaches that amount, it then chooses to propose to the person that meets the threshold or more and it has either previously dated or not.
- For dates in the observe phase if the date proposes we mark them as "proposed" and even if they dont have the highest estimate we propose too. For observe stages we only mark people as "again" if they have given the highest score yet (that doesnt count towards the floor(sqrt(N)) players we need to date in the observe phase)
- In the decide phase we propose to the first player that meets/exceeds the threshold. To choose which players we want to date we check the ones we dated and have >0
- The threshold for the decide phase decreases as rounds progress. We use a decaying value gamma that is calculated using the number of rounds left and the number of singles left: decay= chances/(people-1), where chances=min(rounds_left, singles-1).

### Liar detection

- We keept track of each person's projections and ideals. Lying is shown in various cases: their projection changes between dates (with us), their described ideal matches our projection on the same date exactly or by some % of similarity (use cosine similarity), their described ideal changes between dates, and their projection matches the ideal we described on an earlier date.

### Where to implement

Make these chnages in players/player_2/player.py. You are allowed to create
helper functions inside the player class. The methods advertise(), choose_dates(), describe_ideal(), and decide() cannot be deleted and are the ones needed to be implemented. advertise() is only done at the start and cannot be changed later, choose_dates() is called every round and it tells the system who it filtered to choose the dates from, describe_ideal() can change per round, and decide() is called per date.

### Implementation write-up

Create an .md file outlining what changes you made
