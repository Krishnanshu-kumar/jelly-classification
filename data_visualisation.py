import os
from collections import defaultdict
import matplotlib.pyplot as plt


def count_images(main_folder):
    """
    Count all images in a main folder and its subfolders and display a graph.

    Parameters:
    main_folder (str): Path to the main folder containing images

    Returns:
    tuple: (total count, dictionary of counts per folder)
    """
    # Common image file extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    # Dictionary to store counts for each folder
    folder_counts = defaultdict(int)
    total_count = 0

    # Walk through all folders
    for root, dirs, files in os.walk(main_folder):
        # Count images in current folder
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
        count = len(image_files)

        if count > 0:
            # Get relative path for cleaner output
            rel_path = os.path.relpath(root, main_folder)
            folder_counts[rel_path] = count
            total_count += count

    # Print results
    print(f"\nTotal images found: {total_count}")
    print("\nBreakdown by folder:")
    print("-" * 50)

    for folder, count in folder_counts.items():
        print(f"{folder}: {count} images")

    # Create visualization
    plt.figure(figsize=(12, 6))
    folders = list(folder_counts.keys())
    counts = list(folder_counts.values())

    # Create bar plot
    plt.bar(range(len(folders)), counts, color='skyblue')
    plt.xticks(range(len(folders)), folders, rotation=45, ha='right')
    plt.xlabel('Folders')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Images Across Folders')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Add value labels on top of each bar
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')

    plt.show()

    return total_count, folder_counts


# Example usage
if __name__ == "__main__":
    folder_path = 'dataset'
    total, counts = count_images(folder_path)