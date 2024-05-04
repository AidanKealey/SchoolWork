import pandas as pd
import json

def get_data(state):
    print("get_data -- Start")
    review_file_path = f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review-{state}_10.json'
    df_review_data = pd.read_json(review_file_path, lines=True)

    metadata_file_path = f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/meta-{state}.json'
    df_ratings = pd.read_json(metadata_file_path, lines=True)

    df_ratings = df_ratings[df_ratings['category'].astype(str).str.contains('restaurant', case=False, na=False)]
    df_ratings = df_ratings.drop_duplicates(subset='gmap_id')

    df_all_data = pd.merge(df_review_data, df_ratings, on='gmap_id', how='inner')

    df_all_data = df_all_data.fillna('')
    df_all_data = df_all_data[df_all_data['text'] != '']  # drop reviews with no comments

    df_data = df_all_data[['rating', 'text']].copy()
    df_data = df_data.rename(columns={'rating': 'stars', 'text': 'user_comment'})

    df_data['stars'] = pd.to_numeric(df_data['stars'], errors='coerce', downcast='integer')

    df_data = df_data.dropna(subset=['stars'])
    df_data['user_comment'] = df_data['user_comment'].astype(str)

    print("get_data -- End")

    df_data.to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/processed/filtered_data_{state}.csv', mode='a', index=False)

    return 


def updateCSV(state):
    review_file_path = f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/new/review_sentiments_{state}_stars.csv'
    file_star = pd.read_csv(review_file_path, header=None)

    file_star = file_star.drop(file_star.columns[1], axis=1)
    file_star.to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review_sentiments_stars.csv', mode='a', index=False, header=False)

    review_file_path2 = f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/new/review_sentiments_{state}_geo.csv'
    file_geo = pd.read_csv(review_file_path2, header=None)

    last_column = file_geo.iloc[:, -1]
    file_geo = file_geo.drop(file_geo.columns[0], axis=1)
    file_geo = file_geo.drop(file_geo.columns[0], axis=1)
    file_geo = file_geo.drop(file_geo.columns[-1], axis=1)
    file_geo.insert(0, 'state', last_column)

    file_geo.to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review_sentiments_geo.csv', mode='a', index=False, header=False)

    return


def combineCSVs(state):
    review_file_path = f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/processed/review_sentiments_{state}_stars.csv'
    file_star = pd.read_csv(review_file_path, header=None)

    file_star = file_star.drop(file_star.columns[1], axis=1)
    file_star.to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review_sentiments_stars.csv', mode='a', index=False, header=False)

    review_file_path2 = f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/processed/review_sentiments_{state}_geo.csv'
    file_geo = pd.read_csv(review_file_path2, header=None)

    last_column = file_geo.iloc[:, -1]
    file_geo = file_geo.drop(file_geo.columns[0], axis=1)
    file_geo = file_geo.drop(file_geo.columns[0], axis=1)
    file_geo = file_geo.drop(file_geo.columns[-1], axis=1)
    file_geo.insert(0, 'state', last_column)

    file_geo.to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review_sentiments_geo.csv', mode='a', index=False, header=False)

    return


def exploreData(state):
    #stuff
    print("exploreData -- Start")

    metadata_file_path = f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/meta-{state}.json'
    df_ratings = pd.read_json(metadata_file_path, lines=True)

    numRatings = df_ratings.shape[0]
    print(f'number of businesses: {numRatings}')
    df_ratings = df_ratings[df_ratings['category'].astype(str).str.contains('restaurant', case=False, na=False)]
    df_ratings = df_ratings.drop_duplicates(subset='gmap_id')
    numOfRestaurantRatings = df_ratings.shape[0]
    print(f'number of restaurants: {numOfRestaurantRatings}')

    cols = ['gmap_id', 'rating', 'text', 'pics']
    data = []
    review_file_path = f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review-{state}_10.json'

    with open(review_file_path, encoding='latin-1') as f:
        for line in f:
            doc = json.loads(line)
            lst = [doc['gmap_id'], doc['rating'], doc['text'], doc['pics']]
            data.append(lst)

    df_review_data = pd.DataFrame(data=data, columns=cols)

    df_all_data = pd.merge(df_review_data, df_ratings, on='gmap_id', how='inner')

    del df_review_data, df_ratings

    picture_data_frame = pd.DataFrame(df_all_data, columns=['pics'])
    picture_data_frame['pics'] = picture_data_frame['pics'].apply(lambda x: len(x) if x is not None else 0)
    
    recordsPics = len(picture_data_frame['pics'])
    print(f'number of records for Pics: {recordsPics}')
    pic_counts = picture_data_frame['pics'].value_counts().sort_index()
    zeroPics = pic_counts[0]
    print(f'number of reviews with no picture: {zeroPics}')
    onePlusPics = pic_counts[1:].sum()
    print(f'number of reviews with atleast one picture: {onePlusPics}')

    del picture_data_frame

    df_data_commets = df_all_data[['rating', 'text']].copy()
    
    df_data_commets = df_data_commets.fillna('')
    recordsComment = len(df_data_commets['text'])
    print(f'number commets (including NaN): {recordsComment}')
    df_data_commets = df_data_commets[df_data_commets['text'] != '']
    recordsWithComment = len(df_data_commets['text'])
    print(f'number of commets (without NaN): {recordsWithComment}')
    recordsWithoutComment = recordsComment - recordsWithComment
    print(f'number of NaN commets: {recordsWithoutComment}')

    df_data = df_all_data[['rating', 'text']].copy()
    df_data = df_data.rename(columns={'rating': 'stars', 'text': 'user_comment'})

    df_data['stars'] = pd.to_numeric(df_data['stars'], errors='coerce', downcast='integer')

    recordsStars = len(df_data['stars'])
    print(f'number stars (including NaN): {recordsStars}')

    df_data = df_data.dropna(subset=['stars'])
    df_data['user_comment'] = df_data['user_comment'].astype(str)

    recordsWithStars = len(df_data['stars'])
    print(f'number of stars (without NaN): {recordsWithStars}')
    recordsWithoutStars = recordsStars - recordsWithStars
    print(f'number of NaN stars: {recordsWithoutStars}')

    averageRating = df_data['stars'].mean()
    print(f'avergae rating in state: {state}, {averageRating:.2f} stars')
    star_counts = df_data['stars'].value_counts().sort_index()
    print(f'star distribution in state: {state} \n{star_counts}')

    print("exploreData -- End")

    return numRatings, numOfRestaurantRatings, recordsPics, zeroPics, onePlusPics, recordsComment, recordsWithComment, recordsWithoutComment, recordsStars, recordsWithStars, recordsWithoutStars, averageRating, star_counts



states = ['California', 'Texas', 'New_York']
# states = ['Montana']
# s = 'California'

# for s in states:
#     get_data(s)

# for s in states:
#     combineCSVs(s)

# for s in states:
#     updateCSV(s)

# all_numRatings = 0
# all_numOfRestaurantRatings = 0
# all_recordsPics = 0
# all_zeroPics = 0
# all_onePlusPics = 0
# all_recordsComment = 0
# all_recordsWithComment = 0
# all_recordsWithoutComment = 0
# all_recordsStars = 0
# all_recordsWithStars = 0
# all_recordsWithoutStars = 0
# all_averageRating = 0
# all_star_count_1 = 0
# all_star_count_2 = 0
# all_star_count_3 = 0
# all_star_count_4 = 0
# all_star_count_5 = 0

# for s in states:
#     print(f'state: {s}')

#     numRatings, numOfRestaurantRatings, recordsPics, zeroPics, onePlusPics, recordsComment, recordsWithComment, recordsWithoutComment, recordsStars, recordsWithStars, recordsWithoutStars, averageRating, star_counts = exploreData(s)

#     all_numRatings += numRatings
#     all_numOfRestaurantRatings += numOfRestaurantRatings
#     all_recordsPics += recordsPics
#     all_zeroPics += zeroPics
#     all_onePlusPics += onePlusPics
#     all_recordsComment += recordsComment
#     all_recordsWithComment += recordsWithComment
#     all_recordsWithoutComment += recordsWithoutComment
#     all_recordsStars += recordsStars
#     all_recordsWithStars += recordsWithStars
#     all_recordsWithoutStars += recordsWithoutStars
#     all_averageRating += averageRating

#     all_star_count_1 += star_counts.iloc[0]
#     all_star_count_2 += star_counts.iloc[1]
#     all_star_count_3 += star_counts.iloc[2]
#     all_star_count_4 += star_counts.iloc[3]
#     all_star_count_5 += star_counts.iloc[4]

# print('Totals for all states:')
# print(f'Total number of businesses:                         {all_numRatings}')
# print(f'Total number of restaurants:                        {all_numOfRestaurantRatings}')
# print(f'number of picture records (with/without pics):      {all_recordsPics}')
# print(f'Reviews with 0 pictures:                            {all_zeroPics}')
# print(f'Reviews with 1+ pictures:                           {all_onePlusPics}')
# print(f'number of comment records (with/without comments):  {all_recordsComment}')
# print(f'number of records with comments:                    {all_recordsWithComment}')
# print(f'number of comments without comments:                {all_recordsWithoutComment}')
# print(f'number of star records:                             {all_recordsStars}')
# print(f'number of records with stars:                       {all_recordsWithStars}')
# print(f'number of records without stars:                    {all_recordsWithoutStars}')
# print(f'average star rating across all three states:        {all_averageRating}')
# print(f'number of 1 star records:                           {all_star_count_1}')
# print(f'number of 2 star records:                           {all_star_count_2}')
# print(f'number of 3 star records:                           {all_star_count_3}')
# print(f'number of 4 star records:                           {all_star_count_4}')
# print(f'number of 5 star records:                           {all_star_count_5}')

# data = {
#     "all_numRatings": [all_numRatings],
#     "all_numOfRestaurantRatings": [all_numOfRestaurantRatings],
#     "all_recordsPics": [all_recordsPics],
#     "all_zeroPics": [all_zeroPics],
#     "all_onePlusPics": [all_onePlusPics],
#     "all_recordsComment": [all_recordsComment],
#     "all_recordsWithComment": [all_recordsWithComment],
#     "all_recordsWithoutComment": [all_recordsWithoutComment],
#     "all_recordsStars": [all_recordsStars],
#     "all_recordsWithStars": [all_recordsWithStars],
#     "all_recordsWithoutStars": [all_recordsWithoutStars],
#     "all_averageRating": [all_averageRating],
#     "all_star_count_1": [all_star_count_1],
#     "all_star_count_2": [all_star_count_2],
#     "all_star_count_3": [all_star_count_3],
#     "all_star_count_4": [all_star_count_4],
#     "all_star_count_5": [all_star_count_5]
# }


# df = pd.DataFrame(data)
# print(df)

# df.to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/explore/datav2.csv', index=False)

