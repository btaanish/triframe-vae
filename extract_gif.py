from PIL import Image
import os
from tqdm import tqdm

def extract_frames_from_gif(gif_path, output_folder):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the GIF file
    with Image.open(gif_path) as img:
        # Extract the file name without its extension
        file_name = os.path.splitext(os.path.basename(gif_path))[0]
        
        # Loop over each frame in the GIF
        frame_number = 0
        while True:
            # Try to save the current frame as a PNG
            frame = img.copy()
            frame.save(os.path.join(output_folder, f"{file_name}_frame{frame_number}.png"))
            frame_number += 1
            
            # Move to the next frame
            try:
                img.seek(img.tell() + 1)
            except EOFError:
                break  # Exit the loop when there are no more frames

def process_all_gifs_in_folder(input_folder, output_folder):
    # List all files in the input directory
    bar = tqdm(os.listdir(input_folder))
    for file_name in bar:
        if file_name.lower().endswith('.gif'):
            gif_path = os.path.join(input_folder, file_name)
            extract_frames_from_gif(gif_path, output_folder)

# Example usage
input_folder = '/root/autodl-tmp/SDFusion/test_results_nopre_small_dset/gif'
output_folder = '/root/autodl-tmp/SDFusion/test_results_nopre_small_dset/png'
process_all_gifs_in_folder(input_folder, output_folder)
