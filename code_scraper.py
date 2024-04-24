import os


def scrape_python_files_to_text(directory, output_file):
    """
    Recursively scrapes all Python files in a specified directory and its subdirectories,
    then writes their contents into a single text file.

    Args:
    directory (str): The root directory from which to start scraping Python files.
    output_file (str): The path to the text file where the content of all Python files will be written.
    """
    with open(output_file, 'w') as outfile:
        # Walk through the directory
        for dirpath, dirnames, filenames in os.walk(directory):
            # Filter and process each Python file
            for filename in [f for f in filenames if f.endswith('.py')]:
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, 'r') as file:
                        content = file.read()
                        outfile.write(f"##New # Content from: {filepath}\n")
                        outfile.write(content + "\n\n")
                except IOError as e:
                    print(f"Error reading file {filepath}: {e}")

# Example usage:


scrape_python_files_to_text("/home/julius/ros/ros_ws/src/pycram/src/pycram", 'output.txt')
