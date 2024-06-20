import ee
import pandas as pd
import numpy as np
import geemap.foliumap as geemap
import streamlit as st
from streamlit_folium import st_folium
from google.oauth2 import service_account
from ee import oauth
import os
import matplotlib.pyplot as plt

os.environ["EARTHENGINE_TOKEN"] = st.secrets["EARTHENGINE_TOKEN"]
st.title("Landcover Classification using Sentinel-2 and Dynamic World")

def authenticate_with_service_account():
    service_account_info = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(service_account_info, scopes = oauth.SCOPES)
    ee.Initialize(credentials)

authenticate_with_service_account()

# User input for polygon coordinates
polygon_input = st.sidebar.text_area("Enter Polygon Coordinates (as a list of lists, e.g., [[lon1, lat1], [lon2, lat2], ...])", "", height = 300)
year = st.sidebar.selectbox("Select Year", [2020, 2021, 2022, 2023, 2024])

# Define quarters and filter options
quarters = ['Q1', 'Q2', 'Q3', 'Q4', 'Entire Year']
if year == 2024:
    quarters = ['Q1']
quarter = st.sidebar.selectbox("Select Quarter", quarters)

submit_button = st.sidebar.button("Submit")

# Class mapping
class_mapping = {
    10: 'Trees',
    20: 'Shrubland',
    30: 'Grassland',
    40: 'Cropland',
    50: 'Built-up',
    60: 'Bare / Sparse vegetation',
    70: 'Snow and ice',
    80: 'Permanent water bodies',
    90: 'Herbaceous wetland',
    95: 'Mangroves'
}

# Color palette for the WorldCover classes
palette = [
    '006400',  # Trees
    'ffbb22',  # Shrubland
    'ffff4c',  # Grassland
    'f096ff',  # Cropland
    'fa0000',  # Built-up
    'b4b4b4',  # Bare / Sparse vegetation
    'f0f0f0',  # Snow and Ice
    '0064c8',  # Permanent water bodies
    '0096a0',  # Herbaceous wetland
    '00cf75'   # Mangroves
]

def get_date_range(year, quarter):
    if quarter == 'Q1':
        return f'{year}-01-01', f'{year}-04-01'
    elif quarter == 'Q2':
        return f'{year}-04-01', f'{year}-07-01'
    elif quarter == 'Q3':
        return f'{year}-07-01', f'{year}-10-01'
    elif quarter == 'Q4':
        return f'{year}-10-01', f'{year+1}-01-01'
    else:
        return f'{year}-01-01', f'{year+1}-01-01'

if submit_button and polygon_input:
    try:
        # Convert user input to polygon
        polygon_coords = eval(polygon_input)
        geometry = ee.Geometry.Polygon([polygon_coords])
        start_date, end_date = get_date_range(year, quarter)

        def load_datasets(geometry, start_date, end_date):
            # Load ESA WorldCover dataset
            worldCover = ee.Image(f'ESA/WorldCover/v100/2020')
            
            # Create training points from the WorldCover dataset
            trainingPoints = worldCover.sample(
                region=geometry,
                scale=10,
                numPixels=5000,
                seed=0,
                geometries=True
            )

            # Split the data into training and validation sets
            split = 0.7  # 70% training, 30% validation
            withRandom = trainingPoints.randomColumn('random')
            
            # Filter training points
            trainingSet = withRandom.filter(ee.Filter.lt('random', split))
            
            # Filter validation points
            validationSet = withRandom.filter(ee.Filter.gte('random', split))
            
            # Load Sentinel-2 ImageCollection
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 35)) \
                .filter(ee.Filter.date(start_date, end_date)) \
                .filter(ee.Filter.bounds(geometry)) \
                .select('B.*')
            
            # Create a median composite
            s2composite = s2.median()
            
            # Overlay the point on the image to get training data
            training = s2composite.addBands(worldCover).sampleRegions(
                collection=trainingSet,
                properties=['Map'],
                scale=10
            )
            
            # Train a classifier
            s2Classifier = ee.Classifier.smileRandomForest(50).train(
                features=training,
                classProperty='Map',
                inputProperties=s2composite.bandNames()
            )
            
            # Classify the Sentinel-2 image
            s2classified = s2composite.classify(s2Classifier)
            
            # Load Dynamic World ImageCollection and create a composite
            probabilityBands = [
                'water', 'trees', 'grass', 'flooded_vegetation', 'crops',
                'shrub_and_scrub', 'built', 'bare', 'snow_and_ice'
            ]
            
            dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
                .filter(ee.Filter.date(start_date, end_date)) \
                .filter(ee.Filter.bounds(geometry)) \
                .select(probabilityBands)
            
            dwComposite = dw.median()
            
            # Overlay the point on the image to get training data
            training = dwComposite.sampleRegions(
                collection=trainingSet,
                properties=['Map'],
                scale=10
            )
            
            # Train a classifier
            dwclassifier = ee.Classifier.smileRandomForest(50).train(
                features=training,
                classProperty='Map',
                inputProperties=dwComposite.bandNames()
            )
            
            # Classify the Dynamic World image
            dwclassified = dwComposite.classify(dwclassifier)
            
            return s2composite, s2classified, dwComposite, dwclassified, trainingSet, validationSet

        s2composite, s2classified, dwComposite, dwclassified, trainingSet, validationSet = load_datasets(geometry, start_date, end_date)

        # Display the input composite
        rgbVis = {'min': 0.0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}
        Map = geemap.Map(center=[polygon_coords[0][1], polygon_coords[0][0]], zoom=10)
        Map.centerObject(geometry, 14)
        Map.addLayer(s2composite.clip(geometry), rgbVis, 'S2 Composite (RGB)')

        # Add Sentinel-2 and Dynamic World classification to the map
        Map.addLayer(s2classified.clip(geometry), {'min': 10, 'max': 95, 'palette': palette}, 'S2 Classification')
        Map.addLayer(dwclassified.clip(geometry), {'min': 10, 'max': 95, 'palette': palette}, 'DW Classification')

        Map.addLayerControl()

        # Create and display legend-like image for palette colors and labels
        def create_legend_image():
            fig, ax = plt.subplots(figsize=(0.4, 0.2))  # Much smaller size
            handles = [plt.Rectangle((0,0),1,1, color=f'#{color}') for color in palette]
            labels = class_mapping.values()
            legend = ax.legend(handles, labels, loc='center', title='Land Cover Categories')
            ax.axis('off')
            st.pyplot(fig, use_container_width=False)  # Disable full container width

        def calculate_area_proportions(classified, geometry):
            reducer = ee.Reducer.frequencyHistogram()
            areas = classified.reduceRegion(
                reducer=reducer,
                geometry=geometry,
                scale=10,
                bestEffort=True
            )
            area_proportions = areas.getInfo().get('classification', {})
            areas_named = {class_mapping[int(key)]: value for key, value in area_proportions.items()}
            return areas_named

        # Calculate validation metrics
        def calculate_metrics(s2classified, dwclassified, validationSet):
            # Validation: Sample the classified image using the validation set
            s2Validation = s2classified.sampleRegions(
                collection=validationSet,
                properties=['Map'],
                scale=10
            )

            # Calculate the confusion matrix
            s2ConfusionMatrix = s2Validation.errorMatrix('Map', 'classification')
            s2accuracy = round(s2ConfusionMatrix.accuracy().getInfo(), 4)
            s2precision = s2ConfusionMatrix.producersAccuracy().getInfo()
            s2recall = s2ConfusionMatrix.consumersAccuracy().getInfo()

            # Flatten the nested precision and recall lists and compute the average of non-zero elements
            s2flat_precision = np.array(s2precision).flatten()
            s2flat_recall = np.array(s2recall).flatten()

            s2average_precision = round(np.mean(s2flat_precision[s2flat_precision != 0]), 4)
            s2average_recall = round(np.mean(s2flat_recall[s2flat_recall != 0]), 4)

            s2kappa = round(s2ConfusionMatrix.kappa().getInfo(), 2)

            # Identify classes with highest and lowest precision and recall
            s2highest_precision_class = np.argmax(s2flat_precision)

            # Find the index of the smallest non-zero precision value
            s2non_zero_precision_indices = np.where(s2flat_precision != 0)[0]
            s2lowest_precision_class = s2non_zero_precision_indices[np.argmin(s2flat_precision[s2non_zero_precision_indices])]

            s2highest_recall_class = np.argmax(s2flat_recall)

            # Find the index of the smallest non-zero recall value
            s2non_zero_recall_indices = np.where(s2flat_recall != 0)[0]
            s2lowest_recall_class = s2non_zero_recall_indices[np.argmin(s2flat_recall[s2non_zero_recall_indices])]

            s2class_metrics = {}
            for i, class_name in class_mapping.items():
                s2class_metrics[class_name] = {
                    'Precision': round(s2flat_precision[i], 4),
                    'Recall': round(s2flat_recall[i], 4),
                }

            # Validation: Sample the classified image using the validation set
            dwValidation = dwclassified.sampleRegions(
                collection=validationSet,
                properties=['Map'],
                scale=10
            )

            # Calculate the confusion matrix
            dwConfusionMatrix = dwValidation.errorMatrix('Map', 'classification')
            dwaccuracy = round(dwConfusionMatrix.accuracy().getInfo(), 4)
            dwprecision = dwConfusionMatrix.producersAccuracy().getInfo()
            dwrecall = dwConfusionMatrix.consumersAccuracy().getInfo()

            # Flatten the nested precision and recall lists and compute the average of non-zero elements
            dwflat_precision = np.array(dwprecision).flatten()
            dwflat_recall = np.array(dwrecall).flatten()

            dwaverage_precision = round(np.mean(dwflat_precision[dwflat_precision != 0]), 4)
            dwaverage_recall = round(np.mean(dwflat_recall[dwflat_recall != 0]), 4)

            dwkappa = round(dwConfusionMatrix.kappa().getInfo(), 4)

            # Identify classes with highest and lowest precision and recall
            dwhighest_precision_class = np.argmax(dwflat_precision)

            # Find the index of the smallest non-zero precision value
            dwnon_zero_precision_indices = np.where(dwflat_precision != 0)[0]
            dwlowest_precision_class = dwnon_zero_precision_indices[np.argmin(dwflat_precision[dwnon_zero_precision_indices])]

            dwhighest_recall_class = np.argmax(dwflat_recall)

            # Find the index of the smallest non-zero recall value
            dwnon_zero_recall_indices = np.where(dwflat_recall != 0)[0]
            dwlowest_recall_class = dwnon_zero_recall_indices[np.argmin(dwflat_recall[dwnon_zero_recall_indices])]

            dwclass_metrics = {}
            for i, class_name in class_mapping.items():
                dwclass_metrics[class_name] = {
                    'Precision': round(dwflat_precision[i], 4),
                    'Recall': round(dwflat_recall[i], 4),
                }

            metrics_data = {
                "Metric": ["Accuracy", "Precision", "Recall", "Kappa Coefficient", "Class with Highest Precision", "Class with Lowest Precision", "Class with Highest Recall", "Class with Lowest Recall"],
                "Sentinel-2": [s2accuracy, s2average_precision, s2average_recall, s2kappa, class_mapping[s2highest_precision_class], class_mapping[s2lowest_precision_class], class_mapping[s2highest_recall_class], class_mapping[s2lowest_recall_class]],
                "Dynamic World": [dwaccuracy, dwaverage_precision, dwaverage_recall, dwkappa, class_mapping[dwhighest_precision_class], class_mapping[dwlowest_precision_class], class_mapping[dwhighest_recall_class], class_mapping[dwlowest_recall_class]]
            }
            
            return metrics_data, s2class_metrics, dwclass_metrics

        def plot_area_proportions(s2_area_proportions, dw_area_proportions):
            labels = list(class_mapping.values())
            s2_values = [s2_area_proportions.get(class_name, 0) for class_name in labels]
            dw_values = [dw_area_proportions.get(class_name, 0) for class_name in labels]

            # Normalize values to get proportions
            s2_total = sum(s2_values)
            dw_total = sum(dw_values)
            s2_proportions = [value / s2_total for value in s2_values]
            dw_proportions = [value / dw_total for value in dw_values]

            # Sort values in descending order
            sorted_s2 = sorted(zip(s2_proportions, labels, s2_values), key=lambda x: x[0], reverse=True)
            sorted_dw = sorted(zip(dw_proportions, labels, dw_values), key=lambda x: x[0], reverse=True)
            s2_proportions, s2_labels, s2_values = zip(*sorted_s2)
            dw_proportions, dw_labels, dw_values = zip(*sorted_dw)

            fig, ax = plt.subplots(figsize=(10, 2))

            # Plot Sentinel-2
            s2_left = np.cumsum([0] + list(s2_proportions[:-1]))
            dw_left = np.cumsum([0] + list(dw_proportions[:-1]))
            ax.barh('Sentinel-2', s2_proportions, left=s2_left, color=[f'#{palette[labels.index(label)]}' for label in s2_labels], edgecolor='white')
            ax.barh('Dynamic World', dw_proportions, left=dw_left, color=[f'#{palette[labels.index(label)]}' for label in dw_labels], edgecolor='white')

            # Add proportion labels to bars
            for i, (prop, left) in enumerate(zip(s2_proportions, s2_left)):
                if prop > 0.1:
                    ax.text(left + prop / 2, 'Sentinel-2', f'{prop:.1%}', ha='center', va='center', color='white', fontsize=8)

            for i, (prop, left) in enumerate(zip(dw_proportions, dw_left)):
                if prop > 0.05:
                    ax.text(left + prop / 2, 'Dynamic World', f'{prop:.1%}', ha='center', va='center', color='white', fontsize=8)

            ax.set_xlim(0, 1)
            ax.set_xlabel('Proportion')
            ax.set_yticks(['Sentinel-2', 'Dynamic World'])
            ax.set_yticklabels(['Sentinel-2', 'Dynamic World'])
            ax.set_title('Land Cover Area Proportions')

            plt.tight_layout()
            st.pyplot(fig)

        s2_area_proportions = calculate_area_proportions(s2classified, geometry)
        dw_area_proportions = calculate_area_proportions(dwclassified, geometry)

        metrics_data, s2class_metrics, dwclass_metrics = calculate_metrics(s2classified, dwclassified, validationSet)
        metrics_df = pd.DataFrame(metrics_data)

        # Display class-specific metrics in a table
        class_metrics = []
        not_present_categories = []

        for class_name in class_mapping.values():
            s2_metrics = s2class_metrics.get(class_name, {"Precision": "N/A", "Recall": "N/A"})
            dw_metrics = dwclass_metrics.get(class_name, {"Precision": "N/A", "Recall": "N/A"})

            # Replace 0 values with "N/A"
            s2_precision = "N/A" if s2_metrics["Precision"] == 0 else s2_metrics["Precision"]
            s2_recall = "N/A" if s2_metrics["Recall"] == 0 else s2_metrics["Recall"]
            dw_precision = "N/A" if dw_metrics["Precision"] == 0 else dw_metrics["Precision"]
            dw_recall = "N/A" if dw_metrics["Recall"] == 0 else dw_metrics["Recall"]

            # Check if all metrics are "N/A"
            if s2_precision == "N/A" and s2_recall == "N/A" and dw_precision == "N/A" and dw_recall == "N/A":
                not_present_categories.append(class_name)
            else:
                class_metrics.append([class_name, s2_precision, s2_recall, dw_precision, dw_recall])

        class_metrics_df = pd.DataFrame(class_metrics, columns=["Category", "S2 Precision", "S2 Recall", "DW Precision", "DW Recall"])

        Map.to_streamlit(height=600)

        plot_area_proportions(s2_area_proportions, dw_area_proportions)
        create_legend_image()
            
        st.table(metrics_df)
        st.table(class_metrics_df)

        # Print out categories not present in area
        if not_present_categories:
            st.write("Categories not present in area: ", not_present_categories)

    except Exception as e:
        st.error(f"Error: {e}")
