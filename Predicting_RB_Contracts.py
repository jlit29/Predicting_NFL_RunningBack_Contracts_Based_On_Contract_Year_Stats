# Libraries, all should work as pip install
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Note: Figures will save in the directory of the script

# Get Contracts
contracts = nfl.import_contracts()
contracts = contracts.loc[(contracts['year_signed'] > 2012)]   
rb_contracts = contracts.loc[contracts['position'] == 'RB']

# Get yearly stats
years = []
for i in range(2012,2023):
    years.append(i)

stats = nfl.import_seasonal_data(years)
stats = stats.loc[(stats['rushing_yards'] > 99) & (stats['season_type'] == "REG")]

# Get rosters (to merge player IDs and names)
rosters = nfl.import_rosters(years)
rb_rosters = rosters.loc[(rosters['depth_chart_position'] == 'RB')]
keep = ['player_name','player_id', 'entry_year']
rb_rosters = rb_rosters[keep].drop_duplicates()

# Get merging columns to have same name
rb_contracts = rb_contracts.rename(columns={'player' : 'player_name'})
rb_rosters = rb_rosters.rename(columns={'entry_year' : 'draft_year'})

# Merge
merging = ['player_name', 'draft_year']
rb_df = pd.merge(rb_contracts, rb_rosters, on=merging)

# Contract year_signed, look at stats for year_signed-1
rb_df['season'] = rb_df['year_signed'] - 1
df = rb_df.merge(stats, on=['player_id', 'season'], how='left').dropna(subset=['rushing_yards'])

# "Cleaning" (Get total TDs, Cap %, rename target share)
df['apy_cap_pct'] = df['apy_cap_pct'] * 100
df = df.rename(columns={'target_share': 'target_share_percentage'})
df['total_touchdowns'] = df['rushing_tds'] + df['receiving_tds']

# Check out some relational plots
features = ['rushing_epa', 'rushing_first_downs', 'target_share_percentage', 'receiving_epa', 
            'total_touchdowns']

def customize_axes_and_save(x):
    """
    Function to help customize graph aesthetic (x labels, title, etc.)
    """
    plt.title(f'Average Salary Cap % and {var}')
    plt.ylabel('Average Salary Cap %')
    plt.xlabel(var)
    # Get default x-axis tick locations
    xticks = plt.xticks()[0]
    # Get default x-axis limits
    x_min, x_max = plt.xlim()
    x_min = math.floor(x_min)
    x_max = math.ceil(x_max)
    # Calculate default interval
    default_interval = math.ceil((max(df[x]) - min(df[x])) / len(xticks))
    if default_interval > 10:
        default_interval = 10
    if min(df[x]) < 0:
        plt.xticks(np.arange(round(x_min, -1), round(x_max+default_interval, -1), step=10))
    else:
        plt.xticks(np.arange(0, x_max+default_interval, step=default_interval)) 
    plt.savefig(f'{var}.png', dpi=400,bbox_inches='tight')
        
for x in features:
        var = x.replace('_', ' ').title()
        if 'Epa' in var:
            var = var.replace('Epa', 'EPA')
        sns.lmplot(data=df, x=x, y='apy_cap_pct', height=4, aspect=2, 
                   scatter_kws={'color': 'black', 'edgecolor': 'grey'}, line_kws={'color': 'black'})
        customize_axes_and_save(x)

# Set up df for linear regression ML model
df_clean = df.dropna(subset=(features+['apy_cap_pct']))
df_clean = df_clean.loc[:, features+['apy_cap_pct']]
X = df_clean.drop(['apy_cap_pct'], axis = 1)
y = df_clean['apy_cap_pct']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=29)

# Train
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train).fit()
# Predict
X_test = sm.add_constant(X_test)
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
r = np.sqrt(r2)

# ANOVA
formula = 'apy_cap_pct ~ ' + ' + '.join(X_train.columns[1:])
results = smf.ols(formula=formula, data=df_clean).fit()
anova_table = sm.stats.anova_lm(results, typ=2)

# Plot residuals
residual_df = pd.concat([y_test, y_pred], axis=1)
residual_df = residual_df.rename(columns={
    'apy_cap_pct': 'y_test', 0: 'y_pred'})
residual_df['residuals'] = residual_df['y_test'] - residual_df['y_pred']
sns.relplot(data=residual_df, x = 'y_pred', y = 'residuals',
           height=4, aspect=2, color = 'black', edgecolor='grey')
plt.title('Residual Plot')
plt.xlabel('Predicted Y Values')
plt.ylabel('Residuals')
plt.savefig('residuals.png', dpi=400,bbox_inches='tight')
plt.show()

# Display Metrics
d = {'RMSE': [rmse], 'R-Sqrd': [r2], 'R': [r], 'Model p Value': [results.f_pvalue]}
results_df = pd.DataFrame.from_dict(d, orient='columns')
results_df = results_df.rename(index={0: 'Model'})
print('Model Metrics:')
print(results_df)
print("\nRegression coefficients:")
print(results.params)
print('\nANOVA Table')
print(anova_table)
