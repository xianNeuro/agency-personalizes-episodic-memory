#!/usr/bin/env python3
"""
Data Structure Script for Recall Analysis

This script sets up the data structure for loading recall vectors from:
- Two stories: BeforeAlice (BA) and Monthiversary (MV)
- Three conditions: free, yoke, pasv
- Recall vectors stored in summary Excel files:
  - BA (Adventure): data/adventure_data1.xlsx
    - Sheets: rcl_free22, rcl_yoke45, rcl_pasv49
  - MV (Romance): data/romance_data1.xlsx
    - Sheets: rcl_free100, rcl_yoke53, rcl_pasv55
- Individual subject event and recall files stored in:
  - BA: data/individual_data/adventure/{condition}/
  - MV: data/individual_data/romance/{condition}/
- Story maps stored in:
  - BA: data/adventure_data1.xlsx (sheet: story_map)
  - MV: data/romance_data1.xlsx (sheet: story_map)
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

class RecallDataLoader:
    """Class to handle loading recall data from different stories and conditions"""
    
    def __init__(self, base_path=None):
        """
        Initialize the data loader
        
        Parameters:
        base_path: Base path to the scripts directory (defaults to current directory './')
        """
        if base_path is None:
            # Use current directory if not specified
            self.base_path = os.path.abspath('.')
        else:
            self.base_path = base_path
        
        # Define data directory
        self.data_dir = os.path.join(self.base_path, "data")
        
        # Define story paths for individual data
        self.story_paths = {
            'BA': os.path.join(self.data_dir, "individual_data", "adventure"),
            'MV': os.path.join(self.data_dir, "individual_data", "romance")
        }
        
        # Define conditions
        self.conditions = ['free', 'yoke', 'pasv']
        
        # Condition to folder number mapping
        self.condition_to_number = {
            'free': '1',
            'yoke': '2',
            'pasv': '3'
        }
        
        # Story map file paths
        self.story_map_files = {
            'BA': os.path.join(self.data_dir, "adventure_data1.xlsx"),
            'MV': os.path.join(self.data_dir, "romance_data1.xlsx")
        }
        
        # Recall data file name (for individual files)
        self.recall_filename = "rate-recall.xlsx"
        
        # Output directory structure
        self.output_base = os.path.join(self.base_path, "output")
    
    def get_output_dir(self, script_name):
        """
        Get the output directory for a specific script
        
        Parameters:
        script_name: Name of the script (without .py extension)
        
        Returns:
        Path to the script-specific output directory
        """
        output_dir = os.path.join(self.output_base, script_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def get_story_map(self, story):
        """
        Load the story map from the data file
        
        Parameters:
        story: 'BA' or 'MV'
        
        Returns:
        DataFrame with story map data
        """
        if story not in self.story_map_files:
            raise ValueError(f"Story must be 'BA' or 'MV', got {story}")
        
        map_file = self.story_map_files[story]
        if not os.path.exists(map_file):
            raise FileNotFoundError(f"Story map file not found: {map_file}")
        
        map_df = pd.read_excel(map_file, sheet_name='story_map')
        return map_df
    
    def get_recall_file_path(self, story, condition, subject_id=None):
        """
        Get the path to the recall data file for a specific story and condition
        
        Parameters:
        story: 'BA' or 'MV'
        condition: 'free', 'yoke', or 'pasv'
        subject_id: Subject ID (e.g., 'sub1_4001') - if None, returns directory path
        
        Returns:
        Full path to the rate-recall.xlsx file for a subject, or directory path if subject_id is None
        """
        if story not in self.story_paths:
            raise ValueError(f"Story must be 'BA' or 'MV', got {story}")
        if condition not in self.conditions:
            raise ValueError(f"Condition must be 'free', 'yoke', or 'pasv', got {condition}")
        
        condition_num = self.condition_to_number[condition]
        recall_dir = os.path.join(
            self.story_paths[story],
            f"{condition_num}_{condition}"
        )
        
        if subject_id is None:
            return recall_dir
        
        recall_path = os.path.join(recall_dir, f"{subject_id}_{self.recall_filename}")
        return recall_path
    
    def get_event_data_folder(self, story, condition):
        """
        Get the path to the event data folder for a specific story and condition
        
        Parameters:
        story: 'BA' or 'MV'
        condition: 'free', 'yoke', or 'pasv'
        
        Returns:
        Full path to the event data folder (e.g., data/individual_data/romance/1_free/)
        """
        if story not in self.story_paths:
            raise ValueError(f"Story must be 'BA' or 'MV', got {story}")
        if condition not in self.conditions:
            raise ValueError(f"Condition must be 'free', 'yoke', or 'pasv', got {condition}")
        
        condition_num = self.condition_to_number[condition]
        event_folder = os.path.join(
            self.story_paths[story],
            f"{condition_num}_{condition}"
        )
        
        return event_folder
    
    def get_subject_ids_from_events(self, story, condition):
        """
        Get all subject IDs from event files in the event data folder
        
        Parameters:
        story: 'BA' or 'MV'
        condition: 'free', 'yoke', or 'pasv'
        
        Returns:
        List of subject IDs
        """
        event_folder = self.get_event_data_folder(story, condition)
        
        if not os.path.exists(event_folder):
            raise FileNotFoundError(f"Event data folder not found: {event_folder}")
        
        # Find all files ending with "_events.xlsx"
        event_files = [f for f in os.listdir(event_folder) if f.endswith("_events.xlsx")]
        
        # Extract subject IDs from filenames
        subject_ids = []
        for filename in sorted(event_files):
            # Skip temporary files
            if filename.startswith('~$'):
                continue
            # Remove "_events.xlsx" suffix to get subject ID
            subject_id = filename.replace("_events.xlsx", "")
            subject_ids.append(subject_id)
        
        print(f"Found {len(subject_ids)} subject event files in {event_folder}")
        return subject_ids
    
    def load_subject_event_data(self, story, condition, subject_id):
        """
        Load individual subject event data file
        
        Parameters:
        story: 'BA' or 'MV'
        condition: 'free', 'yoke', or 'pasv'
        subject_id: Subject ID (e.g., 'sub1_4001')
        
        Returns:
        DataFrame with event data
        """
        event_folder = self.get_event_data_folder(story, condition)
        event_file = os.path.join(event_folder, f"{subject_id}_events.xlsx")
        
        if not os.path.exists(event_file):
            raise FileNotFoundError(f"Event file not found: {event_file}")
        
        event_df = pd.read_excel(event_file)
        print(f"Loaded event data for {subject_id}: {event_df.shape[0]} events, {event_df.shape[1]} columns")
        print(f"Columns: {list(event_df.columns)}")
        
        return event_df
    
    def get_choice_and_want_not_info(self, story, condition, subject_id):
        """
        Get choice event information and want_not values for a subject
        
        Parameters:
        story: 'BA' or 'MV'
        condition: 'free', 'yoke', or 'pasv'
        subject_id: Subject ID
        
        Returns:
        Tuple of (is_choice_event, want_not_values) lists
        """
        event_df = self.load_subject_event_data(story, condition, subject_id)
        
        # Check if required columns exist
        if 'scenes' not in event_df.columns:
            raise ValueError(f"'scenes' column not found in event data for {subject_id}")
        
        # Determine choice events (if "_" in scenes column)
        is_choice_event = []
        for scenes_val in event_df['scenes']:
            if pd.isna(scenes_val):
                is_choice_event.append(False)
            else:
                is_choice_event.append('_' in str(scenes_val))
        
        # Get want_not values if column exists
        want_not_values = []
        if 'want_not' in event_df.columns:
            want_not_raw = event_df['want_not'].tolist()
            # Convert to standardized format: 1, 0, or NaN
            for val in want_not_raw:
                if pd.isna(val) or val == '' or val == ' ' or str(val).lower() == 'na':
                    want_not_values.append(np.nan)
                elif val == 1 or val == 1.0 or str(val).strip() == '1':
                    want_not_values.append(1)
                elif val == 0 or val == 0.0 or str(val).strip() == '0':
                    want_not_values.append(0)
                else:
                    want_not_values.append(np.nan)
        else:
            # If want_not column doesn't exist, fill with NaN
            want_not_values = [np.nan] * len(event_df)
        
        return is_choice_event, want_not_values
    
    def load_recall_data(self, story, condition, subject_ids=None):
        """
        Load recall data for a specific story and condition from summary files
        
        Parameters:
        story: 'BA' or 'MV'
        condition: 'free', 'yoke', or 'pasv'
        subject_ids: List of subject IDs to load (if None, loads all subjects)
        
        Returns:
        DataFrame with recall data (columns are subject IDs, rows are events)
        """
        # Map to summary file sheet names
        sheet_mapping = {
            ('BA', 'free'): 'rcl_free22',
            ('BA', 'yoke'): 'rcl_yoke45',
            ('BA', 'pasv'): 'rcl_pasv49',
            ('MV', 'free'): 'rcl_free100',
            ('MV', 'yoke'): 'rcl_yoke53',
            ('MV', 'pasv'): 'rcl_pasv55'
        }
        
        sheet_name = sheet_mapping.get((story, condition))
        if sheet_name is None:
            raise ValueError(f"Unknown story/condition combination: {story}/{condition}")
        
        # Load from data1 file
        data_file = self.story_map_files[story]
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        print(f"Loading recall data from: {data_file} (sheet: {sheet_name})")
        recall_df = pd.read_excel(data_file, sheet_name=sheet_name)
        
        # Filter to requested subjects if provided
        if subject_ids is not None:
            available_cols = [col for col in recall_df.columns if col in subject_ids]
            if len(available_cols) == 0:
                raise ValueError(f"No matching subject IDs found in recall data")
            recall_df = recall_df[available_cols]
        
        print(f"Loaded recall data: {recall_df.shape[0]} events × {recall_df.shape[1]} subjects")
        print(f"Subject IDs: {list(recall_df.columns)}")
        
        return recall_df
    
    def get_subject_recall_vector(self, story, condition, subject_id):
        """
        Get the recall vector for a specific subject
        
        Parameters:
        story: 'BA' or 'MV'
        condition: 'free', 'yoke', or 'pasv'
        subject_id: Subject ID (e.g., 'sub1_4001')
        
        Returns:
        List of recall scores (1.0 = recalled, 0.0 = forgotten, NaN = missing)
        """
        recall_df = self.load_recall_data(story, condition, subject_ids=[subject_id])
        
        if subject_id not in recall_df.columns:
            raise ValueError(f"Subject ID {subject_id} not found in recall data")
        
        recall_vector = recall_df[subject_id].tolist()
        
        # Remove NaN values at the end to get actual length
        # Keep NaN values that are in the middle (actual missing data)
        actual_length = len(recall_vector)
        # Remove trailing NaN values
        while actual_length > 0 and pd.isna(recall_vector[actual_length - 1]):
            actual_length -= 1
        
        recall_vector = recall_vector[:actual_length]
        
        print(f"Subject {subject_id} recall vector length: {len(recall_vector)}")
        print(f"Recalled events (1): {sum(1 for x in recall_vector if x == 1.0)}")
        print(f"Forgotten events (0): {sum(1 for x in recall_vector if x == 0.0)}")
        print(f"Missing events (NaN): {sum(pd.isna(recall_vector))}")
        
        return recall_vector
    
    def plot_recall_vector(self, recall_vector, subject_id, story, condition, script_name="data_structure",
                          is_choice_event=None, want_not_values=None):
        """
        Plot the recall vector as a line plot with choice events and want_not coloring
        
        Parameters:
        recall_vector: List of recall scores
        subject_id: Subject ID for labeling
        story: Story name for labeling
        condition: Condition name for labeling
        script_name: Name of the script (for output directory), default is "data_structure"
        is_choice_event: List of booleans indicating if each event is a choice event (optional)
        want_not_values: List of want_not values (1, 0, or NaN) for coloring (optional)
        """
        # Convert to numpy array for plotting
        recall_array = np.array(recall_vector)
        n_events = len(recall_array)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # If choice event info is provided, grey shade choice events
        if is_choice_event is not None:
            # Create background rectangles for choice events
            for i, is_choice in enumerate(is_choice_event):
                if is_choice:
                    plt.axvspan(i-0.5, i+0.5, alpha=0.3, color='grey', zorder=0)
        
        # Plot line
        plt.plot(recall_array, linewidth=1, color='black', zorder=1)
        
        # Plot markers with want_not coloring if provided
        if want_not_values is not None and len(want_not_values) == n_events:
            for i, (recall_val, want_not) in enumerate(zip(recall_array, want_not_values)):
                if pd.isna(want_not) or want_not == '':
                    # Black for NaN/blank
                    color = 'black'
                elif want_not == 1 or want_not == 1.0:
                    # Green for want_not = 1
                    color = 'green'
                elif want_not == 0 or want_not == 0.0:
                    # Red for want_not = 0
                    color = 'red'
                else:
                    # Default to black
                    color = 'black'
                
                plt.plot(i, recall_val, marker='o', markersize=5, color=color, zorder=2)
        else:
            # Default: plot all markers in black
            plt.plot(recall_array, marker='o', markersize=3, color='black', zorder=2)
        
        plt.xlabel('Event Number', fontsize=12)
        plt.ylabel('Recall Score', fontsize=12)
        plt.title(f'Recall Vector: {subject_id} ({story} - {condition})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 1.1)
        plt.yticks([0, 1], ['Forgotten (0)', 'Recalled (1)'])
        
        # Add legend if want_not coloring is used
        if want_not_values is not None:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='grey', alpha=0.3, label='Choice Event'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                          markersize=8, label='Want (1)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                          markersize=8, label='Not Want (0)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                          markersize=8, label='N/A')
            ]
            plt.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Add statistics
        recalled = np.sum(recall_array == 1.0)
        forgotten = np.sum(recall_array == 0.0)
        total = len(recall_array)
        recall_rate = recalled / total if total > 0 else 0
        
        stats_text = f'Total: {total} | Recalled: {recalled} | Forgotten: {forgotten} | Rate: {recall_rate:.2%}'
        plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=10)
        
        plt.tight_layout()
        
        # Get output directory for this script
        output_dir = self.get_output_dir(script_name)
        
        # Create filename with new naming convention
        filename = f"{subject_id}_{story}_{condition}_recall-vector_line-plot.png"
        output_file = os.path.join(output_dir, filename)
        
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved to: {output_file}")
        
        plt.close()  # Close figure instead of showing it


def main():
    """Demo: Load BA story Passive condition recall vector for first subject"""
    
    # Initialize data loader (uses current directory by default)
    loader = RecallDataLoader()
    
    print("=== Data Structure Setup ===")
    print(f"Base path: {loader.base_path}")
    print(f"Data directory: {loader.data_dir}")
    print(f"Story paths:")
    for story, path in loader.story_paths.items():
        print(f"  {story}: {path}")
    print(f"Conditions: {loader.conditions}")
    print()
    
    # Demo: Load BA story Passive condition
    print("=== Demo: Loading BA Story Passive Condition ===")
    story = 'BA'
    condition = 'pasv'
    
    try:
        # Get subject IDs
        subject_ids = loader.get_subject_ids_from_events(story, condition)
        if len(subject_ids) == 0:
            print("No subjects found")
            return
        
        first_subject_id = subject_ids[0]
        print(f"\nFirst subject ID: {first_subject_id}")
        
        # Get recall vector for first subject
        recall_vector = loader.get_subject_recall_vector(story, condition, first_subject_id)
        
        print(f"\n=== Recall Vector ===")
        print(f"Length: {len(recall_vector)}")
        print(f"First 20 values: {recall_vector[:20]}")
        print(f"Last 20 values: {recall_vector[-20:]}")
        
        # Plot the recall vector
        print(f"\n=== Plotting Recall Vector ===")
        script_name = os.path.splitext(os.path.basename(__file__))[0]  # Get script name without .py
        loader.plot_recall_vector(recall_vector, first_subject_id, story, condition, script_name)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that the file path is correct.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
