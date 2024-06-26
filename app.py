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
st.title("Landcover Classification using Sentinel-2, Dynamic World and ESA WorldCover")

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

def maskS2clouds(image):
    qa = image.select('QA60')
    # Bits 10 and 11 are clouds and cirrus, respectively
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    # Both flags should be set to zero, indicating clear conditions
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask).divide(10000) 

if submit_button and polygon_input:
    # Convert user input to polygon
    polygon_coords = eval(polygon_input)
    geometry = ee.Geometry.Polygon([polygon_coords])
    start_date, end_date = get_date_range(year, quarter)

    def load_datasets(geometry, start_date, end_date):
        # Load ESA WorldCover dataset
        worldCover = ee.Image('ESA/WorldCover/v100/2020').clip(geometry)
        
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
            .map(maskS2clouds) \
            .select(['B2', 'B3', 'B4', 'B8']) 
        
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

        return s2composite, s2classified, dwComposite, dwclassified, worldCover, trainingSet, validationSet

    s2composite, s2classified, dwComposite, dwclassified, worldCover, trainingSet, validationSet = load_datasets(geometry, start_date, end_date)

    # Display the input composite
    rgbVis = {'min': 0.0, 'max': 0.3, 'bands': ['B4', 'B3', 'B2']}
    Map = geemap.Map(center=[polygon_coords[0][1], polygon_coords[0][0]], zoom=10)
    Map.centerObject(geometry, 14)
    Map.addLayer(s2composite.clip(geometry), rgbVis, 'S2 Composite (RGB)')

    # Add Sentinel-2, Dynamic World, and ESA WorldCover classification to the map
    Map.addLayer(worldCover, {'min': 10, 'max': 95, 'palette': palette}, 'ESA WorldCover 2020')
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
        def get_class_metrics(confusionMatrix, class_mapping):
            accuracy = round(confusionMatrix.accuracy().getInfo(), 4)
            precision = confusionMatrix.producersAccuracy().getInfo()
            recall = confusionMatrix.consumersAccuracy().getInfo()

            flat_precision = np.array(precision).flatten()
            flat_recall = np.array(recall).flatten()

            expected_classes = len(class_mapping)
            class_metrics = {}

            for i, class_name in class_mapping.items():
                if i < len(flat_precision):
                    class_metrics[class_name] = {
                        'Precision': round(flat_precision[i], 4) if i < len(flat_precision) else 0,
                        'Recall': round(flat_recall[i], 4) if i < len(flat_recall) else 0,
                    }
                else:
                    class_metrics[class_name] = {
                        'Precision': 0,
                        'Recall': 0,
                    }

            # Calculate average precision and recall, ignoring zeroes
            valid_precisions = [v['Precision'] for v in class_metrics.values() if v['Precision'] != 0]
            valid_recalls = [v['Recall'] for v in class_metrics.values() if v['Recall'] != 0]
            
            average_precision = round(np.mean(valid_precisions), 4) if valid_precisions else 0
            average_recall = round(np.mean(valid_recalls), 4) if valid_recalls else 0

            kappascore = confusionMatrix.kappa().getInfo(), 4
            kappa = round(kappascore, 4) if isinstance(kappascore, (int, float)) else 0

            return accuracy, average_precision, average_recall, kappa, class_metrics

        # Validation: Sample the classified image using the validation set
        s2Validation = s2classified.sampleRegions(
            collection=validationSet,
            properties=['Map'],
            scale=10
        )

        # Calculate the confusion matrix
        s2ConfusionMatrix = s2Validation.errorMatrix('Map', 'classification')

        s2accuracy, s2average_precision, s2average_recall, s2kappa, s2class_metrics = get_class_metrics(s2ConfusionMatrix, class_mapping)

        # Validation: Sample the classified image using the validation set
        dwValidation = dwclassified.sampleRegions(
            collection=validationSet,
            properties=['Map'],
            scale=10
        )

        # Calculate the confusion matrix
        dwConfusionMatrix = dwValidation.errorMatrix('Map', 'classification')

        dwaccuracy, dwaverage_precision, dwaverage_recall, dwkappa, dwclass_metrics = get_class_metrics(dwConfusionMatrix, class_mapping)

        # Identify classes with highest and lowest precision and recall for both S2 and DW
        def get_extreme_classes(class_metrics):
            non_zero_classes = {k: v for k, v in class_metrics.items() if v['Precision'] != 0}
            highest_precision_class = max(non_zero_classes, key=lambda x: non_zero_classes[x]['Precision'])
            lowest_precision_class = min(non_zero_classes, key=lambda x: non_zero_classes[x]['Precision'])
            highest_recall_class = max(non_zero_classes, key=lambda x: non_zero_classes[x]['Recall'])
            lowest_recall_class = min(non_zero_classes, key=lambda x: non_zero_classes[x]['Recall'])
            return highest_precision_class, lowest_precision_class, highest_recall_class, lowest_recall_class

        s2highest_precision_class, s2lowest_precision_class, s2highest_recall_class, s2lowest_recall_class = get_extreme_classes(s2class_metrics)
        dwhighest_precision_class, dwlowest_precision_class, dwhighest_recall_class, dwlowest_recall_class = get_extreme_classes(dwclass_metrics)

        metrics_data = {
            "Metric": ["Accuracy", "Precision", "Recall", "Kappa Coefficient", "Class with Highest Precision", "Class with Lowest Precision", "Class with Highest Recall", "Class with Lowest Recall"],
            "Sentinel-2": [s2accuracy, s2average_precision, s2average_recall, s2kappa, s2highest_precision_class, s2lowest_precision_class, s2highest_recall_class, s2lowest_recall_class],
            "Dynamic World": [dwaccuracy, dwaverage_precision, dwaverage_recall, dwkappa, dwhighest_precision_class, dwlowest_precision_class, dwhighest_recall_class, dwlowest_recall_class]
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

        class_metrics.append([class_name, s2_metrics["Precision"], s2_metrics["Recall"], dw_metrics["Precision"], dw_metrics["Recall"]])

    class_metrics_df = pd.DataFrame(class_metrics, columns=["Category", "S2 Precision", "S2 Recall", "DW Precision", "DW Recall"])

    Map.to_streamlit(height=600)

    plot_area_proportions(s2_area_proportions, dw_area_proportions)
    create_legend_image()
        
    st.table(metrics_df)
    st.table(class_metrics_df)

