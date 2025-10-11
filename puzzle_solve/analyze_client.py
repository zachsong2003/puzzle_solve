#!/usr/bin/env python3
"""
Client program to analyze puzzle pieces using the server's analyze-pieces endpoint.

Usage:
    python analyze_client.py <input_image_file>

Example:
    python analyze_client.py scrambled_puzzle.jpg
"""

import sys
import requests
import json
from pathlib import Path


def analyze_puzzle_pieces(input_image_file, server_url="http://127.0.0.1:8000"):
    """
    Send an image to the server's analyze-pieces endpoint and return the analysis.
    
    Args:
        input_image_file (str): Path to the image file containing scrambled puzzle pieces
        server_url (str): Base URL of the puzzle solver server
        
    Returns:
        dict: Analysis results from the server
    """
    # Check if input file exists
    image_path = Path(input_image_file)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image file not found: {input_image_file}")
    
    # Prepare the request
    analyze_url = f"{server_url}/analyze-pieces"
    
    try:
        # Open and send the image file
        with open(image_path, 'rb') as image_file:
            files = {'image': (image_path.name, image_file, 'image/jpeg')}
            
            print(f"Sending image '{input_image_file}' to {analyze_url}...")
            response = requests.post(analyze_url, files=files)
            
        # Check response status
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            error_msg = f"Server returned error {response.status_code}"
            try:
                error_detail = response.json().get('error', 'Unknown error')
                error_msg += f": {error_detail}"
            except:
                error_msg += f": {response.text}"
            raise Exception(error_msg)
            
    except requests.exceptions.ConnectionError:
        raise Exception(f"Could not connect to server at {server_url}. Make sure the server is running.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {e}")


def print_analysis_results(result):
    """
    Print the analysis results in a formatted way.
    
    Args:
        result (dict): Analysis results from the server
    """
    print("\n" + "="*60)
    print("PUZZLE PIECE ANALYSIS RESULTS")
    print("="*60)
    
    if 'analysis' in result:
        print("\nAI Analysis:")
        print("-" * 40)
        print(result['analysis'])
    
    if 'reference_pieces_used' in result:
        print(f"\nReference pieces used: {result['reference_pieces_used']}")
    
    if 'grid_info' in result:
        grid = result['grid_info']
        print(f"\nGrid structure: {grid['rows']} rows × {grid['columns']} columns")
        print("Expected piece positions:")
        for piece in grid['pieces']:
            print(f"  - Row {piece['row']}, Column {piece['column']}: {piece['file']}")
    
    print("\n" + "="*60)


def main():
    """Main function to handle command line arguments and run the analysis."""
    if len(sys.argv) != 2:
        print("Usage: python analyze_client.py <input_image_file>")
        print("\nExample:")
        print("  python analyze_client.py scrambled_puzzle.jpg")
        print("\nThis program will:")
        print("  1. Send your image to the puzzle solver server")
        print("  2. Analyze the scrambled pieces against reference pieces")
        print("  3. Identify which piece goes in which position")
        sys.exit(1)
    
    input_image_file = sys.argv[1]
    
    try:
        # Analyze the puzzle pieces
        result = analyze_puzzle_pieces(input_image_file)
        
        # Print the results
        print_analysis_results(result)
        
        # Optionally save results to file
        output_file = f"analysis_results_{Path(input_image_file).stem}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults also saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()