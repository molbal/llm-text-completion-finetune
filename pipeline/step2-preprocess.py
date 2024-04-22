import os
import argparse


def process_file(input_file, output_dir):
    # Open the input file
    with open(input_file, 'r', encoding="utf-8") as f:
        # Read the file content
        content = f.read()

        # Find the index of the start and end lines
        start_index = content.find('*** START OF THE PROJECT GUTENBERG EBOOK')
        end_index = content.find('*** END OF THE PROJECT GUTENBERG EBOOK')

        # Check if both start and end lines exist in the file
        if start_index != -1 and end_index != -1:
            # Extract the book content between the start and end lines
            lines = content[start_index:end_index].strip().split('\n')

            # Remove the first and last line
            del lines[0]
            del lines[-1]

            # Join the remaining lines back into a single string
            book_content = '\n'.join(lines)

            # Create the output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Create the output file path
            output_file = os.path.join(output_dir, os.path.basename(input_file))

            # Write the book content to the output file
            with open(output_file, 'w', encoding="utf-8") as out_f:
                out_f.write(book_content)

        else:
            print(f"Skipping {input_file} as it doesn't contain both start and end lines.")


def main():
    parser = argparse.ArgumentParser(description='Process TXT files and extract book content.')
    parser.add_argument('--input_dir', type=str, help='Path to the input directory containing TXT files')
    parser.add_argument('--output_dir', type=str,
                        help='Path to the output directory where the extracted books will be saved')

    args = parser.parse_args()

    # Iterate through all text files in the input directory
    for filename in os.listdir(args.input_dir):
        process_file(os.path.join(args.input_dir, filename), args.output_dir)

    print(f"✅ Done.")


if __name__ == '__main__': main()