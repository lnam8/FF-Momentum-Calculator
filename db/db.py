import sqlite3
import csv
import os

def create_db_tables(DB):
    # Connect to the SQLite database
    conn = sqlite3.connect(DB)

    # Create a cursor object to execute SQLite queries
    cursor = conn.cursor()

    # Madden Weekly table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS madden_weekly (
            year TEXT,
            week TEXT,
            college TEXT,
            signingBonus_diff INTEGER,
            awareness_rating INTEGER,
            shortRouteRunning_diff INTEGER,
            press_diff INTEGER,
            carrying_diff INTEGER,
            strength_rating INTEGER,
            catchInTraffic_rating INTEGER,
            pursuit_rating INTEGER,
            plyrAssetname TEXT,
            breakSack_diff INTEGER,
            plyrPortrait_diff INTEGER,
            catching_rating INTEGER,
            spinMove_rating INTEGER,
            acceleration_diff INTEGER,
            breakTackle_diff INTEGER,
            height INTEGER,
            finesseMoves_rating INTEGER,
            strength_diff INTEGER,
            runBlock_rating INTEGER,
            tackle_rating INTEGER,
            runBlock_diff INTEGER,
            kickPower_diff INTEGER,
            zoneCoverage_rating INTEGER,
            plyrBirthdate TEXT,
            awareness_diff INTEGER,
            runningStyle_rating INTEGER,
            totalSalary INTEGER,
            trucking_rating INTEGER,
            toughness_diff INTEGER,
            hitPower_diff INTEGER,
            tackle_diff INTEGER,
            jukeMove_rating INTEGER,
            playRecognition_rating INTEGER,
            shortRouteRunning_rating INTEGER,
            status TEXT,
            lastName TEXT,
            jerseyNum_diff INTEGER,
            jerseyNum INTEGER,
            breakSack_rating INTEGER,
            passBlockFinesse_diff INTEGER,
            jumping_rating INTEGER,
            throwAccuracyDeep_diff INTEGER,
            stamina_diff INTEGER,
            throwAccuracyShort_diff INTEGER,
            powerMoves_diff INTEGER,
            throwOnTheRun_diff INTEGER,
            zoneCoverage_diff INTEGER,
            jukeMove_diff INTEGER,
            speed_diff INTEGER,
            release_rating INTEGER,
            agility_diff INTEGER,
            hitPower_rating INTEGER,
            throwAccuracyMid_rating INTEGER,
            kickAccuracy_rating INTEGER,
            impactBlocking_diff INTEGER,
            stamina_rating INTEGER,
            plyrPortrait TEXT,
            kickPower_rating INTEGER,
            throwUnderPressure_rating INTEGER,
            team TEXT,
            signingBonus INTEGER,
            height_diff INTEGER,
            playAction_diff INTEGER,
            throwUnderPressure_diff INTEGER,
            changeOfDirection_diff INTEGER,
            blockShedding_rating INTEGER,
            fullNameForSearch TEXT,
            overall_rating INTEGER,
            deepRouteRunning_diff INTEGER,
            passBlockFinesse_rating INTEGER,
            runBlockFinesse_diff INTEGER,
            throwPower_rating INTEGER,
            kickReturn_rating INTEGER,
            leadBlock_rating INTEGER,
            bCVision_rating INTEGER,
            primaryKey_diff INTEGER,
            mediumRouteRunning_diff INTEGER,
            playAction_rating INTEGER,
            totalSalary_diff INTEGER,
            teamId_diff INTEGER,
            leadBlock_diff INTEGER,
            catchInTraffic_diff INTEGER,
            mediumRouteRunning_rating INTEGER,
            acceleration_rating INTEGER,
            spinMove_diff INTEGER,
            yearsPro_diff INTEGER,
            spectacularCatch_rating INTEGER,
            injury_rating INTEGER,
            weight INTEGER,
            playRecognition_diff INTEGER,
            deepRouteRunning_rating INTEGER,
            firstName TEXT,
            yearsPro INTEGER,
            manCoverage_diff INTEGER,
            catching_diff INTEGER,
            throwAccuracyShort_rating INTEGER,
            position TEXT,
            overall_diff INTEGER,
            weight_diff INTEGER,
            bCVision_diff INTEGER,
            throwPower_diff INTEGER,
            speed_rating INTEGER,
            runBlockPower_rating INTEGER,
            injury_diff INTEGER,
            toughness_rating INTEGER,
            throwOnTheRun_rating INTEGER,
            jumping_diff INTEGER,
            spectacularCatch_diff INTEGER,
            manCoverage_rating INTEGER,
            stiffArm_rating INTEGER,
            throwAccuracyMid_diff INTEGER,
            trucking_diff INTEGER,
            passBlock_diff INTEGER,
            powerMoves_rating INTEGER,
            iteration INTEGER,
            stiffArm_diff INTEGER,
            passBlockPower_rating INTEGER,
            impactBlocking_rating INTEGER,
            carrying_rating INTEGER,
            breakTackle_rating INTEGER,
            plyrHandedness TEXT,
            kickReturn_diff INTEGER,
            passBlock_rating INTEGER,
            changeOfDirection_rating INTEGER,
            press_rating INTEGER,
            passBlockPower_diff INTEGER,
            pursuit_diff INTEGER,
            release_diff INTEGER,
            throwAccuracyDeep_rating INTEGER,
            age_diff INTEGER,
            archetype TEXT,
            runBlockPower_diff INTEGER,
            runBlockFinesse_rating INTEGER,
            finesseMoves_diff INTEGER,
            blockShedding_diff INTEGER,
            kickAccuracy_diff INTEGER,
            teamId TEXT,
            agility_rating INTEGER,
            age INTEGER,
            primaryKey INTEGER
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historical_weekly_stats (
            player_name TEXT,
            player_position TEXT,
            player_team TEXT,
            year TEXT,
            week INTEGER,
            receptions INTEGER,
            targets INTEGER,
            receiving_yards INTEGER,
            receiving_yards_per_reception REAL,
            receiving_touchdowns INTEGER,
            rushing_attempts INTEGER,
            rushing_yards INTEGER,
            rushing_yards_per_attempt REAL,
            rushing_touchdowns INTEGER,
            standard_points REAL,
            half_ppr_points REAL,
            ppr_points REAL
        );
    '''
    )

    # Season projections table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS season_projections (
        player_name TEXT,
        player_position TEXT,
        years_experience INTEGER,
        receptions INTEGER,
        receiving_yards INTEGER,
        receiving_touchdowns INTEGER,
        rushing_attempts INTEGER,
        rushing_yards INTEGER,
        rushing_touchdowns INTEGER,
        standard_points REAL,
        half_ppr_points REAL,
        ppr_points REAL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS weekly_projections (
        data_source TEXT,
        player_id INTEGER,
        player_name TEXT,
        player_position TEXT,
        week INTEGER,
        standard_projected_points REAL,
        half_ppr_projected_points REAL,
        ppr_projected_points REAL
    )
    ''')

    # Weekly stats table 
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS weekly_stats (
        player_name TEXT,
        player_position TEXT,
        player_team TEXT,
        week INTEGER,
        receptions INTEGER,
        targets INTEGER,
        receiving_yards INTEGER,
        receiving_yards_per_reception REAL,
        receiving_touchdowns INTEGER,
        rushing_attempts INTEGER,
        rushing_yards INTEGER,
        rushing_yards_per_attempt REAL,
        rushing_touchdowns INTEGER,
        standard_points REAL,
        half_ppr_points REAL,
        ppr_points REAL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS files_processed (
        file_id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT,
        file_dir TEXT
    )
    ''')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()
    print("Finished creating database tables\n")

def delete_db(DB):
    try:
        os.remove(DB)
        print(f"{DB} has been succesfully deleted")

    except Exception as e:
        print(e)

def add_madden_data(DB, csv_file):
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    
    print(f"Starting: Processing Madden weekly file: {csv_file}")

    with open(csv_file, 'r') as f:
        csv_reader = csv.DictReader(f)
    
        for row in csv_reader:
            cursor.execute(f'''
                INSERT INTO madden_weekly (
                    year, week, college, signingBonus_diff, awareness_rating, shortRouteRunning_diff, press_diff,
                    carrying_diff, strength_rating, catchInTraffic_rating, pursuit_rating, plyrAssetname,
                    breakSack_diff, plyrPortrait_diff, catching_rating, spinMove_rating, acceleration_diff,
                    breakTackle_diff, height, finesseMoves_rating, strength_diff, runBlock_rating, tackle_rating,
                    runBlock_diff, kickPower_diff, zoneCoverage_rating, plyrBirthdate, awareness_diff,
                    runningStyle_rating, totalSalary, trucking_rating, toughness_diff, hitPower_diff, tackle_diff,
                    jukeMove_rating, playRecognition_rating, shortRouteRunning_rating, status, lastName,
                    jerseyNum_diff, jerseyNum, breakSack_rating, passBlockFinesse_diff, jumping_rating,
                    throwAccuracyDeep_diff, stamina_diff, throwAccuracyShort_diff, powerMoves_diff, throwOnTheRun_diff,
                    zoneCoverage_diff, jukeMove_diff, speed_diff, release_rating, agility_diff, hitPower_rating,
                    throwAccuracyMid_rating, kickAccuracy_rating, impactBlocking_diff, stamina_rating, plyrPortrait,
                    kickPower_rating, throwUnderPressure_rating, team, signingBonus, height_diff, playAction_diff,
                    throwUnderPressure_diff, changeOfDirection_diff, blockShedding_rating, fullNameForSearch,
                    overall_rating, deepRouteRunning_diff, passBlockFinesse_rating, runBlockFinesse_diff,
                    throwPower_rating, kickReturn_rating, leadBlock_rating, bCVision_rating, primaryKey_diff,
                    mediumRouteRunning_diff, playAction_rating, totalSalary_diff, teamId_diff, leadBlock_diff,
                    catchInTraffic_diff, mediumRouteRunning_rating, acceleration_rating, spinMove_diff, yearsPro_diff,
                    spectacularCatch_rating, injury_rating, weight, playRecognition_diff, deepRouteRunning_rating,
                    firstName, yearsPro, manCoverage_diff, catching_diff, throwAccuracyShort_rating, position,
                    overall_diff, weight_diff, bCVision_diff, throwPower_diff, speed_rating, runBlockPower_rating,
                    injury_diff, toughness_rating, throwOnTheRun_rating, jumping_diff, spectacularCatch_diff,
                    manCoverage_rating, stiffArm_rating, throwAccuracyMid_diff, trucking_diff, passBlock_diff,
                    powerMoves_rating, iteration, stiffArm_diff, passBlockPower_rating, impactBlocking_rating,
                    carrying_rating, breakTackle_rating, plyrHandedness, kickReturn_diff, passBlock_rating,
                    changeOfDirection_rating, press_rating, passBlockPower_diff, pursuit_diff, release_diff,
                    throwAccuracyDeep_rating, age_diff, archetype, runBlockPower_diff, runBlockFinesse_rating,
                    finesseMoves_diff, blockShedding_diff, kickAccuracy_diff, teamId, agility_rating, age, primaryKey
                ) VALUES (
                    :year, :week, :college, :signingBonus_diff, :awareness_rating, :shortRouteRunning_diff, :press_diff,
                    :carrying_diff, :strength_rating, :catchInTraffic_rating, :pursuit_rating, :plyrAssetname,
                    :breakSack_diff, :plyrPortrait_diff, :catching_rating, :spinMove_rating, :acceleration_diff,
                    :breakTackle_diff, :height, :finesseMoves_rating, :strength_diff, :runBlock_rating, :tackle_rating,
                    :runBlock_diff, :kickPower_diff, :zoneCoverage_rating, :plyrBirthdate, :awareness_diff,
                    :runningStyle_rating, :totalSalary, :trucking_rating, :toughness_diff, :hitPower_diff, :tackle_diff,
                    :jukeMove_rating, :playRecognition_rating, :shortRouteRunning_rating, :status, :lastName,
                    :jerseyNum_diff, :jerseyNum, :breakSack_rating, :passBlockFinesse_diff, :jumping_rating,
                    :throwAccuracyDeep_diff, :stamina_diff, :throwAccuracyShort_diff, :powerMoves_diff, :throwOnTheRun_diff,
                    :zoneCoverage_diff, :jukeMove_diff, :speed_diff, :release_rating, :agility_diff, :hitPower_rating,
                    :throwAccuracyMid_rating, :kickAccuracy_rating, :impactBlocking_diff, :stamina_rating, :plyrPortrait,
                    :kickPower_rating, :throwUnderPressure_rating, :team, :signingBonus, :height_diff, :playAction_diff,
                    :throwUnderPressure_diff, :changeOfDirection_diff, :blockShedding_rating, :fullNameForSearch,
                    :overall_rating, :deepRouteRunning_diff, :passBlockFinesse_rating, :runBlockFinesse_diff,
                    :throwPower_rating, :kickReturn_rating, :leadBlock_rating, :bCVision_rating, :primaryKey_diff,
                    :mediumRouteRunning_diff, :playAction_rating, :totalSalary_diff, :teamId_diff, :leadBlock_diff,
                    :catchInTraffic_diff, :mediumRouteRunning_rating, :acceleration_rating, :spinMove_diff, :yearsPro_diff,
                    :spectacularCatch_rating, :injury_rating, :weight, :playRecognition_diff, :deepRouteRunning_rating,
                    :firstName, :yearsPro, :manCoverage_diff, :catching_diff, :throwAccuracyShort_rating, :position,
                    :overall_diff, :weight_diff, :bCVision_diff, :throwPower_diff, :speed_rating, :runBlockPower_rating,
                    :injury_diff, :toughness_rating, :throwOnTheRun_rating, :jumping_diff, :spectacularCatch_diff,
                    :manCoverage_rating, :stiffArm_rating, :throwAccuracyMid_diff, :trucking_diff, :passBlock_diff,
                    :powerMoves_rating, :iteration, :stiffArm_diff, :passBlockPower_rating, :impactBlocking_rating,
                    :carrying_rating, :breakTackle_rating, :plyrHandedness, :kickReturn_diff, :passBlock_rating,
                    :changeOfDirection_rating, :press_rating, :passBlockPower_diff, :pursuit_diff, :release_diff,
                    :throwAccuracyDeep_rating, :age_diff, :archetype, :runBlockPower_diff, :runBlockFinesse_rating,
                    :finesseMoves_diff, :blockShedding_diff, :kickAccuracy_diff, :teamId, :agility_rating, :age, :primaryKey
                );
            ''', row)
    print(f"Finished: Processing file: {csv_file}\n")
    conn.commit()
    conn.close()

def add_historical_weekly_stats(DB, csv_file):

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    
    print(f"Starting: Processing Madden weekly file: {csv_file}")

    with open(csv_file, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            cursor.execute(f'''
                INSERT INTO historical_weekly_stats (
                    player_name, player_position, player_team, year, week, receptions, targets,
                    receiving_yards, receiving_yards_per_reception, receiving_touchdowns, rushing_attempts,
                    rushing_yards, rushing_yards_per_attempt, rushing_touchdowns, standard_points,
                    half_ppr_points, ppr_points
                ) VALUES (
                    :player_name, :player_position, :player_team, :year, :week, :receptions, :targets,
                    :receiving_yards, :receiving_yards_per_reception, :receiving_touchdowns, :rushing_attempts,
                    :rushing_yards, :rushing_yards_per_attempt, :rushing_touchdowns, :standard_points,
                    :half_ppr_points, :ppr_points
                );
            ''', row)

    # Commit the changes and close the database connection
    print(f"Finished: Processing file: {csv_file}\n")
    conn.commit()
    conn.close()

def add_season_projection(DB, csv_file):
    
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    print(f"Starting: Processing season projection file: {csv_file}")

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        # Iterate over each row in the CSV file
        for row in reader:
            # Extract the data from the current row
            player_name = row['player_name']
            player_position = row['player_position']
            years_experience = int(row['years_experience'])
            receptions = int(row['receptions'])
            receiving_yards = int(row['receiving_yards'])
            receiving_touchdowns = int(row['receiving_touchdowns'])
            rushing_attempts = int(row['rushing_attempts'])
            rushing_yards = int(row['rushing_yards'])
            rushing_touchdowns = int(row['rushing_touchdowns'])
            standard_points = float(row['standard_points'])
            half_ppr_points = float(row['half_ppr_points'])
            ppr_points = float(row['ppr_points'])

            # Insert the data into the "season_projections" table
            cursor.execute('''
                INSERT INTO season_projections (
                    player_name, player_position, years_experience,
                    receptions, receiving_yards, receiving_touchdowns,
                    rushing_attempts, rushing_yards, rushing_touchdowns,
                    standard_points, half_ppr_points, ppr_points
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                player_name, player_position, years_experience,
                receptions, receiving_yards, receiving_touchdowns,
                rushing_attempts, rushing_yards, rushing_touchdowns,
                standard_points, half_ppr_points, ppr_points
            ))

    print(f"Finished: Processing file: {csv_file}\n")
    conn.commit()
    conn.close()

def add_weekly_stats(DB, csv_file):

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    print(f"Starting: Processing weekly_stats file: {csv_file}")

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        # Iterate over each row in the CSV file
        for row in reader:
            # Extract the data from the current row
            player_name = row['player_name']
            player_position = row['player_position']
            player_team = row['player_team']
            week = int(row['week'])
            receptions = int(row['receptions'])
            targets = int(row['targets'])
            receiving_yards = int(row['receiving_yards'])
            receiving_yards_per_reception = float(row['receiving_yards_per_reception'])
            receiving_touchdowns = int(row['receiving_touchdowns'])
            rushing_attempts = int(row['rushing_attempts'])
            rushing_yards = int(row['rushing_yards'])
            rushing_yards_per_attempt = float(row['rushing_yards'])
            rushing_touchdowns = int(row['rushing_touchdowns'])
            standard_points = float(row['standard_points'])
            half_ppr_points = float(row['half_ppr_points'])
            ppr_points = float(row['ppr_points'])

            # Insert the data into the "weekly_stats" table
            cursor.execute('''
                INSERT INTO weekly_stats (
                    player_name, player_position, player_team, week,
                    receptions, targets, receiving_yards, receiving_yards_per_reception, receiving_touchdowns,
                    rushing_attempts, rushing_yards, rushing_yards_per_attempt, rushing_touchdowns,
                    standard_points, half_ppr_points, ppr_points
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                player_name, player_position, player_team, week,
                receptions, targets, receiving_yards, receiving_yards_per_reception, receiving_touchdowns,
                rushing_attempts, rushing_yards, rushing_yards_per_attempt, rushing_touchdowns,
                standard_points, half_ppr_points, ppr_points
            ))

    print(f"Finished: Processing file: {csv_file}\n")
    conn.commit()
    conn.close() 

def add_weekly_projections(DB, csv_file):

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    print(f"Starting: Processing weekly_projections file: {csv_file}")

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        # Iterate over each row in the CSV file
        for row in reader:
            # Extract the data from the current row
            data_source = row['data_source']
            player_id = int(row['player_id']) if row['player_id'] else 0 
            player_name = row['player_name']
            player_position = row['player_position']
            week = int(row['week'])
            standard_projected_points = float(row['standard_projected_points']) if row['standard_projected_points'] else 0.00
            half_ppr_projected_points = float(row['half_ppr_projected_points']) if row['half_ppr_projected_points'] else 0.00
            ppr_projected_points = float(row['ppr_projected_points']) if row['ppr_projected_points'] else 0.00

            # Insert the data into the "weekly_stats" table
            cursor.execute('''
                INSERT INTO weekly_projections (
                    data_source, player_id, player_name, player_position, week,
                    standard_projected_points, half_ppr_projected_points, ppr_projected_points
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?) 
            ''', (
                data_source, player_id, player_name, player_position, week,
                standard_projected_points, half_ppr_projected_points, ppr_projected_points
            ))

    print(f"Finished: Processing file: {csv_file}\n")
    conn.commit()
    conn.close() 

def add_processed_file(DB, csv_file, dir):

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO files_processed (file_name, file_dir) VALUES (?, ?)", (csv_file, dir))
    conn.commit()

    cursor.close()
    conn.close()

def process_historical_csv(DB):

    historical_madden_dir = "../scripts/get_historical_madden_ratings/historical_weekly_ratings/"
    historical_stats_dir = "../scripts/get_historical_statistics/"

    files_historical_madden = os.listdir(historical_madden_dir)
    files_historical_stats = os.listdir(historical_stats_dir)

    for file in files_historical_madden:
        full_path = os.path.join(historical_madden_dir, file) 
        add_madden_data(DB, full_path)

    for file in files_historical_stats:
        if ".csv" in file:
            full_path = os.path.join(historical_stats_dir, file) 
            add_historical_weekly_stats(DB, full_path)

def process_season_projection_csv(DB):

    dir = "../scripts/get_season_projections/"

    files = os.listdir(dir)

    for file in files:
        if ".csv" in file:
            full_path = os.path.join(dir, file) 
            add_season_projection(DB, full_path)

def process_madden_ratings(DB): 

    dir = "../scripts/get_madden_ratings/weekly_madden_ratings/"
    
    files = os.listdir(dir)

    for file in files:
        if check_file_exists(DB, file): 
            print(f"{file} already exists. Skipping")
            continue

        if ".csv" in file:
            full_path = os.path.join(dir, file)
            add_madden_data(DB, full_path)
            add_processed_file(DB, file, dir)
            print(f"Added {file} to files_processed")

def process_weekly_stats_csv(DB):

    dir = "../scripts/get_weekly_statistics/"

    files = os.listdir(dir)

    for file in files:
        if ".csv" in file:
            full_path = os.path.join(dir, file) 
            add_weekly_stats(DB, full_path)

def process_weekly_projections_csv(DB):

    dir = "../scripts/get_weekly_projections/"

    files = os.listdir(dir)

    for file in files:
        if ".csv" in file:
            full_path = os.path.join(dir, file) 
            add_weekly_projections(DB, full_path)

def check_file_exists(DB, file_name):
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    # Execute a SELECT statement to check if the file name exists
    cursor.execute("SELECT file_name FROM files_processed WHERE file_name = ?", (file_name,))

    print(f"Searching for file: {file_name}")
    # Fetch the result of the SELECT statement
    result = cursor.fetchone()
    
    file_exists = 0

    # Check if the result is not None, indicating that the file name exists
    if result is not None:
        file_exists = 1

    else:
        file_exists = 0

    # Close the cursor and connection
    cursor.close()
    conn.close()

    return file_exists

def menu(DB):
    while(1):
        print(f"\n{DB} Menu")
        print(f"1. Create {DB}")
        print(f"2. Delete {DB}")
        print("3. Update historical data tables")
        print("4. Update season projection data table")
        print("5. Update Madden weekly ratings data table")
        print("6. Update weekly stats data table")
        print("7. Update weekly projections data table")
        print("8. Reset db and run all scripts")
        print("9. Exit\n")

        choice = input("Enter your choice: ")

        if choice == '1':
            create_db_tables(DB)

        elif choice == '2':
            delete_db(DB)

        elif choice == '3':
            process_historical_csv(DB)

        elif choice == '4':
            process_season_projection_csv(DB)

        elif choice == '5':
            process_madden_ratings(DB)

        elif choice == '6':
            process_weekly_stats_csv(DB)
            
        elif choice == '7':
            process_weekly_projections_csv(DB)
            
        elif choice == '8':
            delete_db(DB)
            create_db_tables(DB)
            process_historical_csv(DB)
            process_season_projection_csv(DB)
            process_madden_ratings(DB)
            process_weekly_stats_csv(DB)
            process_weekly_projections_csv(DB)

        elif choice == '9':
            print("Exiting the program")
            return
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    
    DB = "ff_momentum.db"

    menu(DB)

