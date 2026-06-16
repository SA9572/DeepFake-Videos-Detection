import os
import sys

def show_tree(directory, prefix="", max_depth=3, current_depth=0, 
              ignore_dirs={'.venv', 'venv', '__pycache__', '.git', '.pytest_cache', 
                          '.egg-info', 'node_modules', '.ipynb_checkpoints','DFD_manipulated_sequences', 'DFD_original_sequences'},
              ignore_files={'.DS_Store', '*.pyc', 'thumbs.db','.mp4','.avi','.mkv','.mov','.flv','.wmv','.mp3','.wav','.ogg','.flac'}):
    """
    Display directory tree with smart filtering.
    
    Args:
        directory: Root directory path
        prefix: Tree prefix for display
        max_depth: Maximum depth to traverse
        current_depth: Current recursion depth
        ignore_dirs: Directories to skip
        ignore_files: Files to skip
    """
    
    if max_depth and current_depth >= max_depth:
        return
    
    try:
        items = sorted(os.listdir(directory))
    except PermissionError:
        return
    
    # Filter items
    items = [
        item for item in items 
        if item not in ignore_dirs and not item.startswith('.')
    ]
    
    dirs = [item for item in items if os.path.isdir(os.path.join(directory, item))]
    files = [item for item in items if os.path.isfile(os.path.join(directory, item))]
    
    # Show files
    for i, file in enumerate(files):
        is_last_file = (i == len(files) - 1) and len(dirs) == 0
        connector = "└── " if is_last_file else "├── "
        
        # Get file size
        file_path = os.path.join(directory, file)
        size = os.path.getsize(file_path)
        size_str = format_size(size)
        
        print(f"{prefix}{connector}{file} ({size_str})")
    
    # Show directories
    for i, dir_name in enumerate(dirs):
        is_last = i == len(dirs) - 1
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{dir_name}/")
        
        new_prefix = prefix + ("    " if is_last else "│   ")
        show_tree(os.path.join(directory, dir_name), new_prefix, max_depth, 
                 current_depth + 1, ignore_dirs, ignore_files)

def format_size(bytes):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024
    return f"{bytes:.1f}TB"

def count_items(directory, ignore_dirs):
    """Count files and directories."""
    total_files = 0
    total_dirs = 0
    total_size = 0
    
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        total_dirs += len(dirs)
        total_files += len(files)
        
        for file in files:
            try:
                total_size += os.path.getsize(os.path.join(root, file))
            except:
                pass
    
    return total_files, total_dirs, total_size

# Main execution
if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    
    print(f"\n{'='*60}")
    print(f"📁 Project Structure: {os.path.basename(target_dir)}")
    print(f"📍 Location: {target_dir}")
    print(f"{'='*60}\n")
    
    show_tree(target_dir)
    
    # Show stats
    files, dirs, size = count_items(target_dir, 
        {'.venv', 'venv', '__pycache__', '.git', '.pytest_cache'})
    
    print(f"\n{'='*60}")
    print(f"📊 Summary:")
    print(f"   Directories: {dirs}")
    print(f"   Files: {files}")
    print(f"   Total Size: {format_size(size)}")
    print(f"{'='*60}\n")