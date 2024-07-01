from geopy.distance import geodesic

# Function to calculate total visit duration for a cluster
def calculate_total_time(outlets_df, cluster):
    return sum(outlets_df.loc[outlets_df['id_outlet'].isin(cluster), 'working_hours'])

# Function to find the nearest outlet not yet assigned
def find_nearest_outlet(current_location, unassigned_outlets):
        nearest_outlet = None
        nearest_distance = float('inf')
        for _, outlet in unassigned_outlets.iterrows():
            distance = geodesic(current_location, (outlet['latitude'], outlet['longitude'])).meters
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_outlet = outlet
        return nearest_outlet

def create_cluster(outlets_df, start_outlet, max_time):
        cluster = [start_outlet['id_outlet']]
        current_time = start_outlet['working_hours']
        current_location = (start_outlet['latitude'], start_outlet['longitude'])
        unassigned_outlets = outlets_df[~outlets_df['assigned']]

        while current_time <= max_time:
            nearest_outlet = find_nearest_outlet(current_location, unassigned_outlets)
            if nearest_outlet is None:
                break
            
            if current_time + nearest_outlet['working_hours'] > max_time:
                break
            
            cluster.append(nearest_outlet['id_outlet'])
            current_time += nearest_outlet['working_hours']
            current_location = (nearest_outlet['latitude'], nearest_outlet['longitude'])
            outlets_df.loc[outlets_df['id_outlet'] == nearest_outlet['id_outlet'], 'assigned'] = True
            unassigned_outlets = outlets_df[~outlets_df['assigned']]
        
        return cluster

# Calculate cluster centroids
def calculate_centroids(df):
    return df.groupby('cluster')[['latitude', 'longitude']].mean().to_dict('index')

# Reassign outlets to nearest cluster if they are far from their centroid
def reassign_outliers(df, centroids, max_distance=5000):  # Max distance in meters
        for idx, row in df.iterrows():
            current_cluster = row['cluster']
            centroid = (centroids[current_cluster]['latitude'], centroids[current_cluster]['longitude'])
            distance_to_centroid = geodesic((row['latitude'], row['longitude']), centroid).meters
            
            if distance_to_centroid > max_distance:
                nearest_cluster = None
                nearest_distance = float('inf')
                for cluster_id, centroid_coords in centroids.items():
                    distance = geodesic((row['latitude'], row['longitude']), (centroid_coords['latitude'], centroid_coords['longitude'])).meters
                    if distance < nearest_distance and calculate_total_time(df[df['cluster'] == cluster_id]['id_outlet'].tolist()) + row['working_hours'] <= max_time:
                        nearest_distance = distance
                        nearest_cluster = cluster_id
                
                if nearest_cluster is not None:
                    df.at[idx, 'cluster'] = nearest_cluster
                    centroids = calculate_centroids(df)
        
        return df