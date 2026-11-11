Please now write two simple python module for testing 

1) 
please write a test method, that can run the simulation in headless (no gui) mode, with a list of configurations 

each configurations should be easily manually configured in code as a data class, something like this (also use these values as defaults. the players list are for group 1-10, then greedy and lastly random): 

Run( 
    days = 40,
    marriage_bonus = 200,
    players = [2 for _ in range(12)],
    dims = 5,
    monotonic = 2,
    distribution = 'mixed',
    seed = 4444 
)


after each run, log the result into a csv files. log the final score for each player, which team's code it's running (1 to 10), the day on which they got married (-1 if never married), and who each is married to if any. include header. include a comment line indicating the parameters used on the run. save this csv under player_4/experiment_logs 

2)
please write a simple parsing and analytics script that, takes all the csv logs under player_4/experiment_logs, and calculate the average scoring for each group's players. 