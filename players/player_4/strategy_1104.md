Implement player 4 (players/player_4/player.py), using the following logic: 

(For each of the player’s public actions aka methods:)
advertise():
advertises our most widely applicable attribute / seeks:
For player’s attribute, score monotonic attributes with how high it is (monotonic score of 7 -> score 7), score regular attributes with how centered it is. Advertise the attribute with the highest such score.
Similarly for player’s seek, except monotonic attributes with how low it is  (monotonic score of 7 -> score 10 - 7 = 3). 

Choose_dates()
On first round: 
Rank all persons based on penalty from advertised trait / reverse penalty from advertised seek, express interest in dating the top `self.list_limit` available players. 
On later rounds:
If a person we've dated doesn't match the threshold (threshold of Decide section), remove them from future choose_date lists, but keep them on back burner, until our threshold has lowered over time so that we will request a second date and propose. don't list any player that has become unavailable (married)

Project()
strategy: mildly dishonest to mask extremes
Monotonic attribute: if attribute is <= 3, project 3
Regular: if too extreme adjust to nearest quarter (e.g. if attribute is 10 / 9 -> project 8, if attribute is 0/1 -> project 2) 

Describe_ideal
Project truthfully 

Decide
Calculate the expected score if we marry this date, based on their projected attributes and seeks 
If the person is untruthful, adjust scoring with this person down by 10%. 
Mark this person as untruthful if their projected attribute / seeks are diffeerent from their advertisement
Or if we’ve dated them before and their projected attribute / seeks have changed

return a decision of never_again if expected to score negative (penalty > marriage bonus) This is the only place where we would a decision of never_again

If a person scores with us above a certain threshold, 
If this is the first date, request another date. 
If this is a repeated date, request marriage
The threshold is dynamic based on the number of rounds left. On first 10% of the simulation duration, threshold is 80% of possible scores (aka penalty below 20% of marriage bonus), linearly decreasing to 0% on last 10% of simulation duration (aka as long as penalty < than marriage bonus)
After 2 consecutive failed proposals to the same player, stop requesting repeated dates. In the cool down period, and don’t include them in Choose_dates(). After the cool down period, (if this person is still available), resume including them in Choose_dates()

(implementation strategy)
Keep an internal class of player_profile, one for each other player. keep track of: 
Their advertisement (populated from advertisement)
Their projected attributes and seeks (populated on our last date with them)
Whether they are untruthful (see decide section)  
Our expected marriage scoring with them (calcualted by their projected attributes and seeks, and adjusted by untruthfulness (see decide section) 
Our dating history with them, in a list. Use this to check for cool down.
(keep other player-related info here for clarity)
For people who we’ve dated but don’t match our threshold currently, keep them in a dictionary to express interest in dating later (see Choose_dates – on later rounds)
Implement this as modular / encapsulated as possible. 
Use human-readable and self-explainatory variable names (feel free to use long ones if needed) 
Write plenty of comments, but keep them short and concise 
Especially on math lines / python list comprehension and other hard-to-read lines 

One more clarifying point: please encapsulate the calculation of threshold for easy manual updates. 

---

## Clarifying questions (from Claude, before implementing)

Each question has a **proposed default** in brackets. Edit the answer line, or leave it blank to accept the default.

1. **advertise() – regular-attribute score.** The text says "score with how centered it is", but the example `7 -> (7 - 5) * 2 = 4` measures distance *from* the center (a 5 would score 0, a 10 would score 10). Which one do you mean?
   [Default: centeredness, `score = 10 - |value - 5| * 2`, so 5 -> 10, 7 -> 6, 10 -> 0. Same formula for regular seeks.]
   Answer: score should be highest when the attribute is centered. the previous example was wrong. the centeredness above is correct.

2. **advertise() – value posted.** Should the advertised `own_value` be our true value, or the masked value that `project()` will show? (If they differ, other players doing the same check as us would flag us untruthful.)
   [Default: post the masked/projected value so ad and projection agree. Seek value is our true ideal.]
   Answer: use proposed default. if all our attributes are so extreme that we have to advertise one, use the same mask logic as projection during date 

3. **choose_dates() – ad ranking formula.** How should the two ad penalties combine?
   [Default: same as the happiness formula, `my_penalty + their_penalty / 2`. my_penalty = my ideal vs. their advertised own value; their_penalty = their advertised seek vs. my true profile.]
   Answer: use proposed default

4. **choose_dates() – later rounds, ordering.** On later rounds the list would mix: (a) people we dated who are above the threshold but no mutual repeat was set up, (b) back-burner people whose score now clears the lowered threshold, (c) never-dated people ranked by ad. What order, and do never-dated people keep being listed every round?
   [Default: (a) and (b) together sorted by expected score, best first, then (c) by ad rank, truncated to `list_limit`.]
   Answer: use proposed default. 

5. **decide() – the 10% untruthful penalty.** "Adjust down by 10%" of what? Multiplying the score by 0.9 would make a negative score *less* negative.
   [Default: subtract 10% of the marriage bonus, `score -= 0.1 * bonus`. Once someone is flagged untruthful, the flag stays.]
   Answer: use the stated multiply by 0.9 behavior. A negatively-scoring match would not be considered after the first date anyway, all negative scores are treated the same. 

6. **decide() – what counts as "different" from the ad.** Exact match, or allow some tolerance (others may mask extremes like we do)?
   [Default: exact match, `observed[ad.own_dim] != ad.own_value or stated_ideal[ad.seek_dim] != ad.seek_value`. Same exact check for changes between dates.]
   Answer: exact match. 

7. **decide() – on a repeat date above the threshold**, should we ask for another date as well as proposing (so dating continues if they don't propose back)?
   [Default: yes, `again=True, propose=True`.]
   Answer: Yes, use proposed default

8. **Cool down after 2 failed proposals.** (a) How long is the cool down? (b) If random pairing puts us on a date with them during the cool down, do we still propose if they clear the threshold? (c) After the cool down, does the failed-proposal count reset?
   [Default: (a) 10% of total rounds, at least 1 round; (b) yes, propose but don't ask for another date; (c) yes, it resets.]
   Answer: (a) 5 days. use a static variable for this (b) yes, give decision of propose but not again in this case (c) yes

9. **Partner proposed to us.** If someone proposes to us (`on_outcome` tells us) but they're below our current threshold (score still >= 0), should that change anything?
   [Default: no, ignore it and use the threshold as written.]
   Answer: in this case, if our threshold - our scoring with them is within 10% of marriage bonus, propose to them the next time we date (aka accept some loss for quick exit)


