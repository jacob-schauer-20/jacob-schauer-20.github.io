import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup as soup
from io import StringIO

# source = requests.get('https://masseyratings.com/scores.php?s=604302&sub=11620&all=1&mode=2&format=1')

# webpage = soup(source.content, features='html.parser')

# string = webpage.prettify()
# games_csv_string = StringIO(string)
# games_24_raw = pd.read_csv(games_csv_string, header=None, names=['days', 'date', 'team_1_id', 'team_1_hfa', 'team_1_score', 'team_2_id',
#                                                                   'team_2_hfa',
#                                                                   'team_2_score'])

# teams_24 = pd.read_csv('teams_24.txt', header=None, names=['team_id', 'team'])
# games_1 = pd.merge(games_24_raw, teams_24, left_on='team_1_id', right_on='team_id', how='left')
# games_1 = games_1.drop(['team_1_id', 'team_id'], axis=1)
# games_1 = games_1.rename(columns={'team': 'team_1'})
# games_2 = pd.merge(games_1, teams_24, left_on='team_2_id', right_on='team_id', how='left')
# games_2 = games_2.drop(['team_2_id', 'team_id'], axis=1)
# games_2 = games_2.rename(columns={'team': 'team_2'})
# games_2['season'] = 2025
# games_24 = games_2[['season', 'date', 'team_1', 'team_1_hfa', 'team_1_score', 'team_2', 'team_2_hfa', 'team_2_score']]

# games_24['date'] = pd.to_datetime(games_24['date'].astype(str), format='%Y%m%d')
# min_date = pd.to_datetime(games_24['date'].min())

# for i in range(len(games_24)):
#     games_24.loc[i, 'week'] = int(pd.Timedelta(games_24.loc[i, 'date'] - min_date).days / 7) + 1

# games_prev = pd.read_csv('ncaa_games.csv')
# games_prev['date'] = pd.to_datetime(games_prev['date'])
# games = pd.concat([games_prev, games_24])
# games = games.sort_values(by='date', ascending=True)
# games = games.reset_index(drop=True)

games = pd.read_csv('ncaa_games.csv')
games = games[games['season'] != 2021].reset_index(drop=True)
games['date'] = pd.to_datetime(games['date'])
# %%
team_names = set()
for team in games['team_2'].unique():
    team_names.add(team)
# %%


class NCAATeam:
    def __init__(self, name):
        self.team_name = name
        self.off_rating = 73.5
        self.def_rating = 73.5
        self.rating = 0
        self.wins = 0
        self.losses = 0
        self.off_last_year = 73.5
        self.off_2_year = 73.5
        self.off_3_year = 73.5
        self.def_last_year = 73.5
        self.def_2_year = 73.5
        self.def_3_year = 73.5
# %%


class EloCalculator:

    def __init__(self, hfa=1.25, init_k=.17, k_min=.05, k_week=.01, mov_max=29,
                 mov_min=2, init_k_2012=.21, init_k_2022=.19, k_late=.055):
        self.mov_max = mov_max
        self.mov_min = mov_min
        self.hfa = hfa
        self.init_k = init_k
        self.k_min = k_min
        self.k_week = k_week
        self.init_k_2012 = init_k_2012
        self.init_k_2022 = init_k_2022
        self.k_late = k_late

    def predict_score_1(self, scores, teams):
        if scores['team_1_hfa'] == 0:
            off_1 = teams[scores['team_1']].off_rating
            def_2 = teams[scores['team_2']].def_rating
        elif scores['team_1_hfa'] == 1:
            off_1 = teams[scores['team_1']].off_rating + (self.hfa / 2)
            def_2 = teams[scores['team_2']].def_rating + (self.hfa / 2)
        else:
            off_1 = teams[scores['team_1']].off_rating - (self.hfa / 2)
            def_2 = teams[scores['team_2']].def_rating - (self.hfa / 2)
        return 73.5 + (off_1 - 73.5) + (def_2 - 73.5)

    def predict_score_2(self, scores, teams):
        if scores['team_2_hfa'] == 0:
            off_2 = teams[scores['team_2']].off_rating
            def_1 = teams[scores['team_1']].def_rating
        elif scores['team_2_hfa'] == 1:
            off_2 = teams[scores['team_2']].off_rating + (self.hfa / 2)
            def_1 = teams[scores['team_1']].def_rating + (self.hfa / 2)
        else:
            off_2 = teams[scores['team_2']].off_rating - (self.hfa / 2)
            def_1 = teams[scores['team_1']].def_rating - (self.hfa / 2)
        return 73.5 + (off_2 - 73.5) + (def_1 - 73.5)

    # def win_probability(self, scores, teams):
    #     if scores['team_1_hfa'] == 0:
    #         rating_1 = teams[scores['team_1']].rating
    #         rating_2 = teams[scores['team_2']].rating
    #     elif scores['team_1_hfa'] == 1:
    #         rating_1 = teams[scores['team_1']].rating + self.hfa
    #         rating_2 = teams[scores['team_2']].rating - self.hfa
    #     else:
    #         rating_1 = teams[scores['team_1']].rating - self.hfa
    #         rating_2 = teams[scores['team_2']].rating + self.hfa
    #     return 1 / (1 + (np.exp(.155 * (rating_1 - rating_2))))

    def get_k(self, scores, teams):
        k = self.init_k - (self.k_week * scores['week'])
        j = self.init_k_2012 - (self.k_week * scores['week'])
        x = self.init_k_2022 - (self.k_week * scores['week'])
        if ((scores['season'] == 2012) & (j < self.k_min)):
            return self.k_min
        elif scores['season'] == 2012:
            return j
        elif ((scores['season'] == 2022) & (x < self.k_min)):
            return self.k_min
        elif scores['season'] == 2022:
            return x
        elif scores['week'] >= 16:
            return self.k_late
        elif k < self.k_min:
            return self.k_min
        else:
            return k

    def update_single_game(self, scores, teams):
        if scores['team_1_hfa'] == 0:
            def_1 = teams[scores['team_1']].def_rating
            off_2 = teams[scores['team_2']].off_rating
            def_2 = teams[scores['team_2']].def_rating
            off_1 = teams[scores['team_1']].off_rating
        elif scores['team_1_hfa'] == 1:
            def_1 = teams[scores['team_1']].def_rating - (self.hfa / 2)
            off_2 = teams[scores['team_2']].off_rating - (self.hfa / 2)
            def_2 = teams[scores['team_2']].def_rating + (self.hfa / 2)
            off_1 = teams[scores['team_1']].off_rating + (self.hfa / 2)
        else:
            def_1 = teams[scores['team_1']].def_rating + (self.hfa / 2)
            off_2 = teams[scores['team_2']].off_rating + (self.hfa / 2)
            def_2 = teams[scores['team_2']].def_rating - (self.hfa / 2)
            off_1 = teams[scores['team_1']].off_rating - (self.hfa / 2)

        if scores['season'] == 2025:
            teams[scores['team_1']].wins += 1
            teams[scores['team_2']].losses += 1

        if (scores['team_1_score'] - scores['team_2_score']) > self.mov_max:
            team_1_score = scores['team_1_score'] - (((scores['team_1_score'] - scores['team_2_score']) - self.mov_max)
                                                     / 2)
            team_2_score = scores['team_2_score'] + (((scores['team_1_score'] - scores['team_2_score']) - self.mov_max)
                                                     / 2)
        elif (scores['team_1_score'] - scores['team_2_score']) < self.mov_min:
            team_1_score = scores['team_1_score'] + ((self.mov_min - (scores['team_1_score'] - scores['team_2_score']))
                                                     / 2)
            team_2_score = scores['team_2_score'] - ((self.mov_min - (scores['team_1_score'] - scores['team_2_score']))
                                                     / 2)
        else:
            team_1_score = scores['team_1_score']
            team_2_score = scores['team_2_score']

        teams[scores['team_1']].off_rating += (self.get_k(scores, teams) *
                                               (team_1_score - (73.5 + (off_1 - 73.5) + (def_2 - 73.5))))
        teams[scores['team_2']].off_rating += (self.get_k(scores, teams) *
                                               (team_2_score - (73.5 + (off_2 - 73.5) + (def_1 - 73.5))))
        teams[scores['team_1']].def_rating += (self.get_k(scores, teams) *
                                               (team_2_score - (73.5 + (off_2 - 73.5) + (def_1 - 73.5))))
        teams[scores['team_2']].def_rating += (self.get_k(scores, teams) *
                                               (team_1_score - (73.5 + (off_1 - 73.5) + (def_2 - 73.5))))

        teams[scores['team_1']].rating = teams[scores['team_1']].off_rating - teams[scores['team_1']].def_rating
        teams[scores['team_2']].rating = teams[scores['team_2']].off_rating - teams[scores['team_2']].def_rating


# %%
teams = {}
for team in team_names:
    teams[team] = NCAATeam(team)
elo = EloCalculator()
n_games = games.shape[0]
seasons = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2022, 2023, 2024, 2025]
# %%
for year in seasons:
    for team in teams:
        teams[team].off_3_year = teams[team].off_2_year
        teams[team].off_2_year = teams[team].off_last_year
        teams[team].off_last_year = teams[team].off_rating
        teams[team].def_3_year = teams[team].def_2_year
        teams[team].def_2_year = teams[team].def_last_year
        teams[team].def_last_year = teams[team].def_rating
        if year >= 2014:
            teams[team].off_rating = (.075 * teams[team].off_3_year) + \
                (.85 * teams[team].off_last_year) + (.075 * teams[team].off_2_year)
            teams[team].def_rating = (.075 * teams[team].def_3_year) + \
                (.85 * teams[team].def_last_year) + (.075 * teams[team].def_2_year)
            teams[team].rating = teams[team].off_rating - teams[team].def_rating
    for i in range(n_games):
        if games.loc[i, 'season'] == year:
            games.loc[i, 'error'] = np.abs((games.loc[i, 'team_1_score'] - games.loc[i, 'team_2_score']) -
                                           (elo.predict_score_1(games.iloc[i], teams) - elo.predict_score_2(games.iloc[i], teams)))
            games.loc[i, 'k'] = elo.get_k(games.iloc[i], teams)
            games.loc[i, 'pace'] = teams[games.loc[i, 'team_1']].off_rating + teams[games.loc[i, 'team_1']
                                                                                    ].def_rating + teams[games.loc[i, 'team_2']].off_rating + teams[games.loc[i, 'team_2']].def_rating
            elo.update_single_game(games.iloc[i], teams)
# %%
ratings = pd.DataFrame(columns=['team', 'record', 'off', 'def', 'rating'])
for team in teams.keys():
    ratings.loc[team] = pd.Series(
        {'team': team,
         'record': str(teams[team].wins) + '-' + str(teams[team].losses),
         'off': teams[team].off_rating,
         'def': teams[team].def_rating,
         'rating': teams[team].rating})
teams_24 = pd.read_csv('teams_24.txt', header=None, names=['team_id', 'team'])
ratings = pd.merge(teams_24, ratings, on='team', how='left')
ratings = ratings.drop('team_id', axis=1)
conferences = pd.read_csv('d3_conference.csv')
ratings = pd.merge(ratings, conferences, on='team')
ratings = ratings.sort_values(by='rating', ascending=False)
ratings = ratings.reset_index(drop=True)
ratings['rank'] = ratings.index + 1
ratings = ratings.set_index('rank', drop=True)
ratings = ratings[['team', 'conference', 'record', 'rating', 'off', 'def']]
ratings[['rating', 'off', 'def']] = ratings[['rating', 'off', 'def']].astype(float)

ratings.to_csv('d3_ratings.csv')
# %%
# rmse = pd.DataFrame()

# for j in [.055, .05, .045]:
#     for k in [.055, .06, .065]:
#         teams = {}
#         for team in team_names:
#             teams[team] = NCAATeam(team)
#         elo = EloCalculator(hfa=1.25, init_k=.17, k_min=.05, k_week=.01, mov_max=29,
#                             mov_min=2, init_k_2012=.21, init_k_2022=.19, k_late=.055)
#         error = 0
#         for year in seasons:
#             for team in teams:
#                 teams[team].off_3_year = teams[team].off_2_year
#                 teams[team].off_2_year = teams[team].off_last_year
#                 teams[team].off_last_year = teams[team].off_rating
#                 teams[team].def_3_year = teams[team].def_2_year
#                 teams[team].def_2_year = teams[team].def_last_year
#                 teams[team].def_last_year = teams[team].def_rating
#                 if year >= 2014:
#                     teams[team].off_rating = (.075 * teams[team].off_3_year) + \
#                         (.85 * teams[team].off_last_year) + (.075 * teams[team].off_2_year)
#                     teams[team].def_rating = (.075 * teams[team].def_3_year) + \
#                         (.85 * teams[team].def_last_year) + (.075 * teams[team].def_2_year)
#                     teams[team].rating = teams[team].off_rating - teams[team].def_rating
#             for i in range(n_games):
#                 if games.loc[i, 'season'] == year:
#                     error += np.abs((elo.predict_score_1(games.iloc[i], teams) - elo.predict_score_2(
#                         games.iloc[i], teams)) - (games.loc[i, 'team_1_score'] - games.loc[i, 'team_2_score']))
#                     elo.update_single_game(games.iloc[i], teams)
#         avg_error = error / n_games
#         rmse.loc[j, k] = avg_error
#         # 9.47865658235007
# %%
source_fut = requests.get('https://masseyratings.com/scores.php?s=604302&sub=11620&all=1&mode=2&sch=on&format=1')

webpage_fut = soup(source_fut.content, features='html.parser')

string_fut = webpage_fut.prettify()
StringIO_fut = StringIO(string_fut)
fut_games = pd.read_csv(StringIO_fut, header=None, names=['days', 'date', 'team_1_id', 'team_1_hfa', 'team_1_score', 'team_2_id',
                                                          'team_2_hfa', 'team_2_score'])
fut_games = fut_games[fut_games['team_1_score'] == 0]

games_1_f = pd.merge(fut_games, teams_24, left_on='team_1_id', right_on='team_id', how='left')
games_1_f = games_1_f.drop(['team_1_id', 'team_id'], axis=1)
games_1_f = games_1_f.rename(columns={'team': 'team_1'})
games_2_f = pd.merge(games_1_f, teams_24, left_on='team_2_id', right_on='team_id', how='left')
games_2_f = games_2_f.drop(['team_2_id', 'team_id'], axis=1)
games_2_f = games_2_f.rename(columns={'team': 'team_2'})
games_24_f = games_2_f[['date', 'team_1', 'team_1_hfa', 'team_1_score', 'team_2', 'team_2_hfa', 'team_2_score']]
games_24_f['date'] = pd.to_datetime(games_24_f['date'].astype(str), format='%Y%m%d')
# %%
for i in range(len(games_24_f)):
    if games_24_f.loc[i, 'team_1_hfa'] == -1:
        games_24_f.loc[i, 'hfa'] = '@'
    else:
        games_24_f.loc[i, 'hfa'] = 'neu'
    games_24_f.loc[i, 'team_1_pred_score'] = elo.predict_score_1(games_24_f.iloc[i], teams)
    games_24_f.loc[i, 'team_2_pred_score'] = elo.predict_score_2(games_24_f.iloc[i], teams)

games_24_f['spread'] = games_24_f['team_2_pred_score'] - games_24_f['team_1_pred_score']

games_24_f = games_24_f[['date', 'team_1', 'team_1_pred_score', 'hfa', 'team_2',
                         'team_2_pred_score', 'spread']]
games_24_f.to_csv('future_games.csv', index=False)
