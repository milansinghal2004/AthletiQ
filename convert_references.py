import os
import imageio
from tqdm import tqdm

def convert_avi_to_mp4(input_path, output_path):
    print(f"Converting {os.path.basename(input_path)} -> {os.path.basename(output_path)}...")
    try:
        reader = imageio.get_reader(input_path)
        fps = reader.get_meta_data().get('fps', 25.0)
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
        
        for frame in tqdm(reader, desc="Processing frames", leave=False):
            writer.append_data(frame)
            
        writer.close()
        reader.close()
        return True
    except Exception as e:
        print(f"  Error converting {input_path}: {e}")
        return False

def main():
    ref_dir = os.path.join(os.getcwd(), "assets", "references")
    if not os.path.exists(ref_dir):
        print(f"Directory not found: {ref_dir}")
        return

    avi_files = [f for f in os.listdir(ref_dir) if f.endswith(".avi")]
    
    for avi in avi_files:
        in_path = os.path.join(ref_dir, avi)
        out_path = os.path.join(ref_dir, avi.replace(".avi", ".mp4"))
        
        if convert_avi_to_mp4(in_path, out_path):
            # Optionally remove the old .avi to save space, but we'll keep it for now as backup
            # os.remove(in_path) 
            pass

if __name__ == "__main__":
    main()
