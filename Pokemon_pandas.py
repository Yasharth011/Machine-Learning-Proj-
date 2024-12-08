import pandas as pd

df = pd.read_csv('Pokemon.csv')

#read data till a header/row
df.head(1)

#read Headers
df.columns

#read each coloumn
df[['Name', 'Type 1']]

#read each row
df.iloc[1]

#read a specific location (R,C)
df.iloc[2,1]

# Note : iloc - integer location

# iterate thru rows
for index,rows in df.iterrows():
    index, rows['Name']

# read data using non-integer variables 
df.loc[df['Type 1']=='Fire']

# to get statistical data 
df.describe

#to sort values
df.sort_values('Name', ascending=False)
df.sort_values(['Type 1', 'HP'], ascending=[1,0]) #Type 1 is ascending and HP is descending

# add & remove a column 
df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + df['Sp. Def'] + df['Speed']
#alternate method do above
#df['Total'] = df.loc[:,4:10].sum(axis=1)

df = df.drop(columns=['Total'])

# to make a csv or excel from a data frame
df.to_csv('modified.csv', index=False, sep=' ') # index=False to remove serial no. column
# or df.to_excel

# Filtering data
new_df = df.loc[(df['Type 1'] == 'Grass') & (df['Type 2']=='Poison') & (df['HP'] > 70)] 

# Note : in Pandas for condition we use & sign and not 'and' like in python in general 

# to reset indexing in filtered data 
new_df.reset_index(drop=True, inplace=True) # inplace=True allows to modify to the same variable rather stroing to another variable 

df.loc[df['Name'].str.contains('Mega')]

# Conditional Changes
df.loc[df['Type 1']=='Fire', 'Legendary'] = True

# Aggregate Statistics 
# df.groupby(['Type 1', 'Type 2']).mean()/sum()/count()

# Dealing with big data frame 
df = pd.read_csv('modified.csv', chunksize = 5)


