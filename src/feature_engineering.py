import pandas as pd
import numpy as np
from config import TRAIN_FILE, TEST_FILE, TRAIN_CLEANED, TEST_CLEANED, N_REGION_CLUSTERS, TRAIN_CLEANED_UNSCALED, TEST_CLEANED_UNSCALED
from helpers import load_data, save_data
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def keyword_flag(text, keywords):
    return bool(any(word in str(text).lower() for word in keywords))

def preprocess_data(df):
    """Perform data cleaning and feature engineering."""

    # Handle outlier values
    df = handle_outlier_price(df)
    df = handle_outlier_sqfeet(df)
    df = handle_outlier_beds_and_baths(df)

    # Handle missing values
    df.dropna(subset=["description"], inplace=True)
    df["lat"] = df.groupby("region")["lat"].transform(lambda x: x.fillna(x.median()))
    df["long"] = df.groupby("region")["long"].transform(lambda x: x.fillna(x.median()))
    df["laundry_options"] = df["laundry_options"].fillna(df["laundry_options"].mode()[0])
    df["parking_options"] = df["parking_options"].fillna(df["parking_options"].mode()[0])

    # Feature Engineering
    df["luxury_features"] = (df["electric_vehicle_charge"] + df["wheelchair_access"] + df["comes_furnished"])

    coords = df[['lat', 'long']]
    kmeans = KMeans(n_clusters=10, random_state=42)
    df['geo_cluster'] = kmeans.fit_predict(coords)

    region_freq = df['region'].value_counts(normalize=True)
    df['region_freq'] = df['region'].map(region_freq)

    df["log_price"] = np.log1p(df["price"])
    df["log_sqfeet"] = np.log1p(df["sqfeet"])

    feature_keywords = {
        "has_outdoor": ["tennis", "court", "business center", "park", "community", "clubhouse"],
        "has_pool": ["pool", "swimming"],
        "has_fitness_center": ["gym", "fitness center"],
        "mentions_price": ["rent", "deposit", "fee", "monthly"],
        "has_maintenance": ["emergency", "maintenance", "onsite"],
        "has_management": ["management", "managed", "professionally", "office"],
        "has_garbage_disposal": ["garbage", "disposal"],
        "has_closet": ["closet", "closets", "storage"],
        "has_kitchen": ["dining", "kitchen", "refrigerator"],
        "has_fan": ["fan", "air"],
        "has_offer": ["offer", "offers", "discount"],
        "has_appliances": ["appliances", "dryer", "laundry", "washer"]
    }

    for feature, words in feature_keywords.items():
        df[feature] = df['description'].apply(lambda x: keyword_flag(x, words))

    # One-hot encoding categorical variables
    df = pd.get_dummies(df, columns=["laundry_options", "parking_options"], drop_first=True)

    type_counts = df["type"].value_counts()
    rare_types = type_counts[type_counts < 1000].index
    df["type"] = df["type"].replace(rare_types, "other")
    type_encoded_train = pd.get_dummies(df["type"], prefix="type")
    df = pd.concat([df, type_encoded_train], axis=1)
    df.drop(columns=["type"], inplace=True)
    
    # clustering regions
    region_stats = df.groupby("region").agg({ "sqfeet": "mean", "lat": "mean", "long": "mean" }).reset_index()
    kmeans = KMeans(n_clusters=N_REGION_CLUSTERS, random_state=42)
    region_stats["region_cluster"] = kmeans.fit_predict(region_stats[["sqfeet", "lat", "long"]])
    df = df.merge(region_stats[["region", "region_cluster"]], on="region", how="left")
    df.drop(columns=["region_cluster_x"], inplace=True, errors="ignore") 
    df.drop(columns=["region_cluster_y"], inplace=True, errors="ignore") 
    # one hot region cluster values
    region_cluster_encoded = pd.get_dummies(df["region_cluster"], prefix="region_cluster")
    df.drop(columns=["region_cluster"], inplace=True)
    df = pd.concat([df, region_cluster_encoded], axis=1)
    valuable = set([2])
    for i in range(10):
        if i in valuable: continue
        df.drop(columns=["region_cluster_"+str(i)], inplace=True)    
    
    state_encoded = pd.get_dummies(df["state"], prefix="state")
    df = pd.concat([df, state_encoded], axis=1)
    
    significant_states = set(['ca', 'co', 'dc', 'hi', 'ks', 'ma'])
    for col in df.columns:
        if col.startswith('state_'):
            if col[-2:] in significant_states: continue
            df.drop(columns=[col], inplace=True)    
    
    # scaling features
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['int32', 'int64', 'float64']).columns
    num_cols = num_cols.drop('price')
    num_cols = num_cols.drop('log_price')
    df_unscaled = df.copy()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # dropping features
    df.drop(columns=["region"], inplace=True)
    df.drop(columns=["description"], inplace=True)
    df.drop(columns=["state"], inplace=True)
    df.drop(columns=["sqfeet"], inplace=True)
    df.drop(columns=["cats_allowed"], inplace=True)
    df.drop(columns=["dogs_allowed"], inplace=True)
    df.drop(columns=["wheelchair_access"], inplace=True)
    df.drop(columns=["comes_furnished"], inplace=True)
    df.drop(columns=["mentions_price"], inplace=True)
    df.drop(columns=["has_pool"], inplace=True)
    df.drop(columns=["has_garbage_disposal"], inplace=True)
    df.drop(columns=["has_fan"], inplace=True)
    df.drop(columns=["has_offer"], inplace=True)
    df.drop(columns=["has_appliances"], inplace=True)
    df.drop(columns=["has_outdoor"], inplace=True)
    df.drop(columns=["type_townhouse"], inplace=True)
    df.drop(columns=["type_other"], inplace=True)
    df.drop(columns=["type_condo"], inplace=True, errors="ignore") 
    df.drop(columns=["type_manufactured"], inplace=True, errors="ignore") 
    df.drop(columns=["type_duplex"], inplace=True, errors="ignore") 
    df.drop(columns=["parking_options_valet parking"], inplace=True)
    df.drop(columns=["parking_options_street parking"], inplace=True)
    df.drop(columns=["parking_options_no parking"], inplace=True)
    df.drop(columns=["laundry_options_no laundry on site"], inplace=True)
    df.drop(columns=["has_closet"], inplace=True)
    df.drop(columns=["has_management"], inplace=True)
    
    df_unscaled.drop(columns=["region"], inplace=True)
    df_unscaled.drop(columns=["description"], inplace=True)
    df_unscaled.drop(columns=["state"], inplace=True)
    df_unscaled.drop(columns=["sqfeet"], inplace=True)
    df_unscaled.drop(columns=["cats_allowed"], inplace=True)
    df_unscaled.drop(columns=["dogs_allowed"], inplace=True)
    df_unscaled.drop(columns=["wheelchair_access"], inplace=True)
    df_unscaled.drop(columns=["comes_furnished"], inplace=True)
    df_unscaled.drop(columns=["mentions_price"], inplace=True)
    df_unscaled.drop(columns=["has_pool"], inplace=True)
    df_unscaled.drop(columns=["has_garbage_disposal"], inplace=True)
    df_unscaled.drop(columns=["has_fan"], inplace=True)   
    df_unscaled.drop(columns=["has_offer"], inplace=True)
    df_unscaled.drop(columns=["has_appliances"], inplace=True)
    df_unscaled.drop(columns=["has_outdoor"], inplace=True)
    df_unscaled.drop(columns=["type_townhouse"], inplace=True)
    df_unscaled.drop(columns=["type_other"], inplace=True)
    df_unscaled.drop(columns=["type_condo"], inplace=True, errors="ignore") 
    df_unscaled.drop(columns=["type_manufactured"], inplace=True, errors="ignore") 
    df_unscaled.drop(columns=["type_duplex"], inplace=True, errors="ignore") 
    df_unscaled.drop(columns=["parking_options_valet parking"], inplace=True)
    df_unscaled.drop(columns=["parking_options_street parking"], inplace=True)
    df_unscaled.drop(columns=["parking_options_no parking"], inplace=True)
    df_unscaled.drop(columns=["laundry_options_no laundry on site"], inplace=True)
    df_unscaled.drop(columns=["has_closet"], inplace=True)
    df_unscaled.drop(columns=["has_management"], inplace=True)

    return df, df_unscaled

def handle_outlier_price(df):
    df_log_price = np.log1p(df["price"])
    Q1_log = df_log_price.quantile(0.25)
    Q3_log = df_log_price.quantile(0.75)
    IQR_log = Q3_log - Q1_log
    lower_bound_log = Q1_log - (3 * IQR_log)
    upper_bound_log = Q3_log + (3 * IQR_log)
    lower_bound = np.expm1(lower_bound_log)
    upper_bound = np.expm1(upper_bound_log)
    df = df[(df["price"] >= lower_bound) & (df["price"] <= upper_bound)]
    df = df[(df["price"] > 0)]
    return df	

def handle_outlier_sqfeet(df):
    df = df[df["sqfeet"] > 0]
    df_log_sqfeet = np.log1p(df["sqfeet"])
    Q1_sqfeet = df_log_sqfeet.quantile(0.25)
    Q3_sqfeet = df_log_sqfeet.quantile(0.75)
    IQR_sqfeet = Q3_sqfeet - Q1_sqfeet
    lower_bound_sqfeet = Q1_sqfeet - (3 * IQR_sqfeet)
    upper_bound_sqfeet = Q3_sqfeet + (3 * IQR_sqfeet)
    lower_bound_sqfeet = np.expm1(lower_bound_sqfeet)
    upper_bound_sqfeet = np.expm1(upper_bound_sqfeet)
    df = df[(df["sqfeet"] >= lower_bound_sqfeet) & (df["sqfeet"] <= upper_bound_sqfeet)]
    return df

def handle_outlier_beds_and_baths(df):
    df = df[(df["beds"] > 0)]
    df = df[(df["beds"] < 1000)]
    df = df[df["baths"] < 25]
    return df

if __name__ == "__main__":
    # Load data
    train_df = load_data(TRAIN_FILE)
    test_df = load_data(TEST_FILE)

    # Preprocess
    train_df, train_df_unscaled = preprocess_data(train_df)
    test_df, test_df_unscaled = preprocess_data(test_df)

    # expected_columns = [col for col in train_df.columns if col.startswith("state_") or col.startswith("type_")]
    # test_df = test_df.reindex(columns=expected_columns, fill_value=False)

    print('Train shape: ', train_df.shape)
    print('test shape: ', test_df.shape)
    print('train info: ', train_df.info())
    print('test info: ', test_df.info())
    

    # Save processed data
    save_data(train_df, TRAIN_CLEANED)
    save_data(test_df, TEST_CLEANED)
    save_data(train_df_unscaled, TRAIN_CLEANED_UNSCALED)
    save_data(test_df_unscaled, TEST_CLEANED_UNSCALED)

    # print("Feature Engineering Completed.")
