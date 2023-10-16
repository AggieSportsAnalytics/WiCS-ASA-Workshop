# WNBA Prediction Workshop

### Step by step tutorial

Install packages

```py
% pip install pandas
% pip install scikit-learn
```

## Part 1: Cleaning our data

Import pandas
```py
import pandas as pd
```

Read our schedule CSV file and remove the unnecessary columns.

```py
# https://www.basketball-reference.com/wnba/years/2023_games.html
schedule = pd.read_csv("reg_season.csv")

# Remove first and last 2 columns (date, unnamed, and notes)
schedule = schedule.iloc[:, 1:-2]
# Display dataframe
schedule.head()
```

Read our stats CSV file and remove the unnecessary columns.

```py
# https://www.basketball-reference.com/wnba/years/2023.html
advanced_stats = pd.read_csv("advanced_stats.csv")

# Remove nan columns
advanced_stats = advanced_stats.dropna(axis=1, how='all')
# Remove fist and last columns (rank, and arena)
advanced_stats = advanced_stats.iloc[:, 1:-1]
# Display dataframe
advanced_stats.head()
```

Combine our dataframes such that the stats align with the teams playing.

```py
# Combine the two data frames by team name
df = pd.merge(schedule, advanced_stats, left_on="Visitor/Neutral", right_on="Team")
df = pd.merge(df, advanced_stats, left_on="Home/Neutral", right_on="Team")

# Remove duplicate columns
df = df.drop(['Team_x', 'Team_y'], axis=1)
# Display dataframe
df.head()
```

Loop through each game and determine which team won. In the `Home_Winner` column, put a `1` if the home team won, if not, put `0`.

```py
# Add a column for to show if the home team won
for index, row in df.iterrows():
    # Determine which team had more points
    if df.loc[index, 'PTS'] > df.loc[index, 'PTS.1']:
        # Place 0 for home loss
        df.loc[index, 'Home_Winner'] = 0
    else:
        # Place 1 for home win
        df.loc[index, 'Home_Winner'] = 1

# Display dataframe
df.head()
```

Remove the columns that which we don't want for our prediction. The model cannot know who won.

```py
# Determine which columns we don't want to use for our prediction model

# Columns we want to remove
remove_cols = ["Home/Neutral", "Visitor/Neutral", "Home_Winner", "PTS", "PTS.1"]

# Columns we want to keep
selected_cols = [x for x in df.columns if x not in remove_cols]

# Display columns we want to keep
df[selected_cols].head()
```

Scale our data. We want our data to be between `0` and `1` for our `Logistic Regression` (used later). We are using `MinMaxScaler` from `sklearn`.

```py
# Scale our data
from sklearn.preprocessing import MinMaxScaler

# Initialize scaler
scalar = MinMaxScaler()
# Scale our data
df[selected_cols] = scalar.fit_transform(df[selected_cols])
# Display dataframe
df.head()
```

## Part 2: Determine our predictors

Using a ridge regression classifier we can initialize a feature selector to use for creating our model.

```py
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier

# Initialize our ridge regression classification
rr = RidgeClassifier(alpha=1.0)

# Initialize our feature selector which picks the best 10 features backward
sfs = SequentialFeatureSelector(rr, n_features_to_select=10, direction='backward')
```

Now use our selector to pick the 10 best features.

```py
# Determine which columns are the most impactful when predicting the winner
sfs.fit(df[selected_cols], df['Home_Winner'])
```

Now we can see which features it selected.

```py
# Create a list of the most impactful columns
predictors = list(df[selected_cols].columns[sfs.get_support()])
# Display the most impactful columns
df[predictors].head()
```

## Part 3: Creating and testing our model

Import the necessary packages.

```py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

Now we can write a function to train and test our model. We will be writing this as a function in order to get the average accuracy based on many simulations.

```py
def monte_carlo(n):
    accuracy = []
    for i in range(n):
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(df[predictors], df['Home_Winner'], test_size=0.2)

        # Train a logistic regression model on the training data
        model = LogisticRegression()
        # Fit the model to our training data
        model.fit(X_train, y_train)

        # Predict the winners for the test data
        y_pred = model.predict(X_test)

        # Evaluate the accuracy of the model on the test data
        accuracy.append(accuracy_score(y_test, y_pred))

    # Get the average accuracy
    score = sum(accuracy) / len(accuracy)
    return score
```

Run our function 1000 times in order to determine the average accuracy of our model.

```py
score = monte_carlo(1000)
print(f"Accuracy: {score}")
```

Output: `Accuracy: 0.7045416666666663`

Based on 1000 simulations, our model is 70% accurate in predicting outcomes of WNBA games.

## Part 4: Predicting the Finals

First, we must only train our model on games where the Las Vegas Aces and New York Liberty do NOT face each other. This would be cheating and our model would not be accurate to predict future games.

```py
# Remove the rows where the Aces and Liberty play against each other
non_aces_liberty_game = df
non_aces_liberty_game = non_aces_liberty_game.drop(non_aces_liberty_game[(non_aces_liberty_game['Home/Neutral'] == 'Las Vegas Aces') & (non_aces_liberty_game['Visitor/Neutral'] == 'New York Liberty')].index)
non_aces_liberty_game = non_aces_liberty_game.drop(non_aces_liberty_game[(non_aces_liberty_game['Home/Neutral'] == 'New York Liberty') & (non_aces_liberty_game['Visitor/Neutral'] == 'Las Vegas Aces')].index)
non_aces_liberty_game[(non_aces_liberty_game['Home/Neutral'] == 'Las Vegas Aces') & (non_aces_liberty_game['Visitor/Neutral'] == 'New York Liberty')]
```

Then, we grab the data for the matchup between the two teams.

```py
# Get a game where the Aces are home and Liberty is away
final_matchup = df[(df['Home/Neutral'] == 'Las Vegas Aces') & (df['Visitor/Neutral'] == 'New York Liberty')][:1]
# Show the predictors we will use
final_matchup[predictors]
```

We can now train our model, based on non Aces v Liberty games, and use it to predict the outcome game we selected.

```py
# Predict the winner of the final matchup
model = LogisticRegression()
model.fit(non_aces_liberty_game[predictors], non_aces_liberty_game['Home_Winner'])

# Predict the outcome of the final_matchup
y_pred = model.predict(final_matchup[predictors])
print(f"Prediction: {y_pred[0]}")
```

Output: `Prediction: 1.0`

This predicts that the home team will win, meaning the Aces take the win for the first game of the finals.

For a more complex prediction for each finals game, reference `predict.ipynb`.

By predicting all potential matchups based on home and away we predict:

| Home                | Away                | Winner           |   
| ------------------- | ------------------- | ---------------- |
| Las Vegas Aces      | New York Liberty    | Home (Aces)   |
| Las Vegas Aces      | New York Liberty    | Home (Aces)   |
| New York Liberty    | Las Vegas Aces      | Home (Liberty) |
| New York Liberty    | Las Vegas Aces      | Home (Liberty) |
| Las Vegas Aces      | New York Liberty    | Home (Aces)   |
