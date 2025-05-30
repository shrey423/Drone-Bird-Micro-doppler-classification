import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

def analyze_dataset(dataset_path):
    """
    Analyze the DIAT-μSAT dataset for class distribution and balance.
    
    Args:
        dataset_path (str): Path to the dataset directory
    """
    # Get all class directories
    class_dirs = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d))]
    
    # Initialize dictionary to store class counts
    class_counts = {}
    
    # Count images in each class
    for class_dir in class_dirs:
        class_path = os.path.join(dataset_path, class_dir)
        image_count = len([f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
        class_counts[class_dir] = image_count
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'Class': list(class_counts.keys()),
        'Count': list(class_counts.values())
    })
    
    # Sort by count
    df = df.sort_values('Count', ascending=False)
    
    # Calculate statistics
    total_images = df['Count'].sum()
    mean_images = df['Count'].mean()
    std_images = df['Count'].std()
    max_images = df['Count'].max()
    min_images = df['Count'].min()
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total number of images: {total_images}")
    print(f"Number of classes: {len(class_counts)}")
    print(f"Mean images per class: {mean_images:.2f}")
    print(f"Standard deviation: {std_images:.2f}")
    print(f"Maximum images in a class: {max_images}")
    print(f"Minimum images in a class: {min_images}")
    
    # Calculate class imbalance ratio
    imbalance_ratio = max_images / min_images
    print(f"Class imbalance ratio (max/min): {imbalance_ratio:.2f}")
    
    # Set style for plots
    plt.style.use('default')
    
    # 1. Bar plot of class distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Class', y='Count')
    plt.title('Class Distribution in DIAT-μSAT Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()
    
    # 2. Pie chart of class proportions
    plt.figure(figsize=(10, 10))
    plt.pie(df['Count'], labels=df['Class'], autopct='%1.1f%%')
    plt.title('Class Proportions in DIAT-μSAT Dataset')
    plt.axis('equal')
    plt.savefig('class_proportions.png')
    plt.close()
    
    # 3. Box plot of class distribution
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df['Count'])
    plt.title('Distribution of Images per Class')
    plt.ylabel('Number of Images')
    plt.savefig('class_distribution_boxplot.png')
    plt.close()
    
    # Save statistics to CSV
    df.to_csv('dataset_statistics.csv', index=False)
    
    # Calculate and print class weights for balanced training
    class_weights = compute_class_weights(df['Count'])
    print("\nRecommended class weights for balanced training:")
    for class_name, weight in zip(df['Class'], class_weights):
        print(f"{class_name}: {weight:.3f}")
    
    return df, class_weights

def compute_class_weights(class_counts):
    """
    Compute class weights for balanced training.
    
    Args:
        class_counts (array): Array of image counts per class
    
    Returns:
        array: Class weights
    """
    total = class_counts.sum()
    weights = total / (len(class_counts) * class_counts)
    return weights

if __name__ == "__main__":
    # Path to the dataset
    dataset_path = "DIAT-uSAT_dataset"
    
    # Analyze dataset
    df, class_weights = analyze_dataset(dataset_path)
    
    print("\nAnalysis complete. Visualizations have been saved.")
    print("Files generated:")
    print("1. class_distribution.png - Bar plot of class distribution")
    print("2. class_proportions.png - Pie chart of class proportions")
    print("3. class_distribution_boxplot.png - Box plot of class distribution")
    print("4. dataset_statistics.csv - Detailed statistics in CSV format") 