#!/usr/bin/env python3
"""
Data Summary Read Script
Loads and explores individual subject's summarized data from Excel files
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from data_structure import RecallDataLoader

def explore_excel_files():
    """Explore the structure of Excel files in the data folder"""
    
    # Use current directory
    base_path = os.path.abspath('.')
    data_dir = os.path.join(base_path, "data")
    
    # Files for Adventure (BA) and Romance (MV) stories
    ba_files = ['adventure_data1.xlsx', 'adventure_data2.xlsx']
    mv_files = ['romance_data1.xlsx', 'romance_data2.xlsx']
    
    print("="*80)
    print("EXPLORING EXCEL FILES STRUCTURE")
    print("="*80)
    
    all_info = {}
    
    # Explore BA files
    print("\n" + "="*80)
    print("ADVENTURE STORY (BA) FILES")
    print("="*80)
    for filename in ba_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"\n{filename}:")
            print("-" * 80)
            xl_file = pd.ExcelFile(filepath)
            print(f"  Sheets: {xl_file.sheet_names}")
            
            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(filepath, sheet_name=sheet_name)
                print(f"\n  Sheet '{sheet_name}':")
                print(f"    Shape: {df.shape}")
                print(f"    Columns: {list(df.columns)}")
                print(f"    First few rows:")
                print(df.head(3).to_string())
                
                if filename not in all_info:
                    all_info[filename] = {}
                all_info[filename][sheet_name] = {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'sample': df.head(3)
                }
    
    # Explore MV files
    print("\n" + "="*80)
    print("ROMANCE STORY (MV) FILES")
    print("="*80)
    for filename in mv_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"\n{filename}:")
            print("-" * 80)
            xl_file = pd.ExcelFile(filepath)
            print(f"  Sheets: {xl_file.sheet_names}")
            
            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(filepath, sheet_name=sheet_name)
                print(f"\n  Sheet '{sheet_name}':")
                print(f"    Shape: {df.shape}")
                print(f"    Columns: {list(df.columns)}")
                print(f"    First few rows:")
                print(df.head(3).to_string())
                
                if filename not in all_info:
                    all_info[filename] = {}
                all_info[filename][sheet_name] = {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'sample': df.head(3)
                }
    
    return all_info

def load_yoked_percent_want():
    """Load percent_want data for yoked condition from both stories"""
    
    # Use current directory
    base_path = os.path.abspath('.')
    data_dir = os.path.join(base_path, "data")
    
    # Load BA (Adventure) - from adventure_data2.xlsx, sheet 'yoke45'
    ba_filepath = os.path.join(data_dir, "adventure_data2.xlsx")
    ba_data = np.array([])
    if os.path.exists(ba_filepath):
        try:
            df_ba = pd.read_excel(ba_filepath, sheet_name='yoke45')
            if 'prcnt_want' in df_ba.columns:
                ba_data = df_ba['prcnt_want'].dropna().values
                print(f"Loaded {len(ba_data)} BA yoked subjects from adventure_data2.xlsx (yoke45)")
        except Exception as e:
            print(f"Error loading BA data: {e}")
    
    # Load MV (Romance) - from romance_data2.xlsx, sheet 'yoke53'
    mv_filepath = os.path.join(data_dir, "romance_data2.xlsx")
    mv_data = np.array([])
    if os.path.exists(mv_filepath):
        try:
            df_mv = pd.read_excel(mv_filepath, sheet_name='yoke53')
            if 'prcnt_want' in df_mv.columns:
                mv_data = df_mv['prcnt_want'].dropna().values
                print(f"Loaded {len(mv_data)} MV yoked subjects from romance_data2.xlsx (yoke53)")
        except Exception as e:
            print(f"Error loading MV data: {e}")
    
    return ba_data, mv_data

def plot_percent_want_distribution(ba_data, mv_data):
    """Plot distribution of percent_want for yoked condition, side by side"""
    
    # Use current directory
    loader = RecallDataLoader()
    output_dir = loader.get_output_dir("data_summary_read")
    
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot Adventure Story (BA)
    axes[0].hist(ba_data, bins=20, edgecolor='black', linewidth=2, alpha=0.7, color='skyblue')
    axes[0].set_xlabel('Percent Want', fontsize=38)
    axes[0].set_ylabel('Frequency', fontsize=38)
    axes[0].set_title('Adventure Story', fontsize=48)
    axes[0].tick_params(axis='both', labelsize=32)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot Romance Story (MV)
    axes[1].hist(mv_data, bins=20, edgecolor='black', linewidth=2, alpha=0.7, color='lightcoral')
    axes[1].set_xlabel('Percent Want', fontsize=38)
    axes[1].set_ylabel('Frequency', fontsize=38)
    axes[1].set_title('Romance Story', fontsize=48)
    axes[1].tick_params(axis='both', labelsize=32)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    filename = "yoked_percent_want_distribution.png"
    output_file = os.path.join(output_dir, filename)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nDistribution plot saved to: {output_file}")
    
    plt.close()

def main():
    """Main function"""
    
    # First, explore the Excel files to understand their structure
    print("\n" + "="*80)
    print("STEP 1: EXPLORING DATA FILES")
    print("="*80)
    all_info = explore_excel_files()
    
    # Then load yoked percent_want data
    print("\n" + "="*80)
    print("STEP 2: LOADING YOKED PERCENT_WANT DATA")
    print("="*80)
    ba_data, mv_data = load_yoked_percent_want()
    
    print(f"\nAdventure Story (BA) - Yoked percent_want:")
    print(f"  Number of subjects: {len(ba_data)}")
    print(f"  Mean: {np.mean(ba_data):.3f}")
    print(f"  Std: {np.std(ba_data):.3f}")
    print(f"  Min: {np.min(ba_data):.3f}, Max: {np.max(ba_data):.3f}")
    
    print(f"\nRomance Story (MV) - Yoked percent_want:")
    print(f"  Number of subjects: {len(mv_data)}")
    print(f"  Mean: {np.mean(mv_data):.3f}")
    print(f"  Std: {np.std(mv_data):.3f}")
    print(f"  Min: {np.min(mv_data):.3f}, Max: {np.max(mv_data):.3f}")
    
    # Create distribution plot
    print("\n" + "="*80)
    print("STEP 3: CREATING DISTRIBUTION PLOT")
    print("="*80)
    plot_percent_want_distribution(ba_data, mv_data)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()

