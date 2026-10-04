import os
import shutil

def move_files_by_keyword(src_dir, dst_dir, keywords, case_insensitive=False):
    """
    Move files from src_dir to dst_dir if their filename contains any of the given keywords.
    
    Parameters:
        src_dir (str): Source directory path.
        dst_dir (str): Destination directory path.
        keywords (list[str]): List of keywords to look for in filenames.
        case_insensitive (bool): If True, keyword matching ignores case.
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Normalize keywords for case-insensitive matching if needed
    normalized_keywords = [k.lower() for k in keywords] if case_insensitive else keywords

    moved_files = []

    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)

        # Skip directories
        if not os.path.isfile(src_path):
            continue

        name_to_check = filename.lower() if case_insensitive else filename

        # Check if filename contains any of the keywords
        if any(keyword in name_to_check for keyword in normalized_keywords):
            dst_path = os.path.join(dst_dir, filename)
            shutil.move(src_path, dst_path)
            moved_files.append(filename)
            print(f"Moved: {filename}")

    print(f"\nTotal files moved: {len(moved_files)}")
    return moved_files

def file_diff(src_path, dst_path):
    source_files = set(os.listdir(src_path))
    dst_files = set(os.listdir(dst_path))
    diff = source_files - dst_files
    if len(diff):
        print(f'Diff found! Diff count: {len(diff)}, filename: {", ".join(diff)}')
    print('complete')

# Example usage
if __name__ == "__main__":
    source_folder = r"F:\\AI\\DDSP-SVC\\raw04"
    destination_folder = r"F:\\AI\\DDSP-SVC\\results_raw04"
    file_diff(source_folder, destination_folder)
    keyword_list = ["元流之子",
 "元流之子",]
#  "妲己",
#  "王昭君",
#  "米莱狄",
#  "镜"]

    # move_files_by_keyword(source_folder, destination_folder, keyword_list)
