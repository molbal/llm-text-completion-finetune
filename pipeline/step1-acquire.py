import argparse
import os
import time
import requests
import re


def clean_filename(input_string):
    allowed_chars = r'[a-zA-Z .-]'
    output_string = re.sub(r'[^' + allowed_chars + ']', '', input_string)
    return output_string


def download_book(book_url, output_dir, filename):
    # Clean filename
    filename = clean_filename(filename)

    # Skip if it already exists
    output_file = os.path.join(output_dir, f"{filename}.txt")
    if os.path.exists(output_file):
        print(f"File {output_file} already exists, skipping")
        return

    # Download the book content
    response = requests.get(book_url)
    response.raise_for_status()
    time.sleep(1)  # Let's not hammer the API

    # Save the book content to a TXT file
    with open(output_file, 'wb') as f:
        f.write(response.content)

    print(f"Downloaded {output_dir}/{filename}.txt")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Query and save books from Project Gutenberg')
    parser.add_argument('--output_dir', required=True, help='Output directory to save books')
    parser.add_argument('--topic', required=True, help='Topic of books e.g. horror')
    parser.add_argument('--num_records', type=int, required=True, help='Number of records to retrieve')

    args = parser.parse_args()
    print(f"Querying '{args.topic}' topic books for download into {args.output_dir}")

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Base URL for querying Project Gutenberg
    base_url = "https://gutendex.com/books/"

    # Query parameters
    params = {
        "languages": "en",
        "copyright": "false",
        "topic": args.topic,
        "mime-type": "text/plain",
    }

    # Send the request and get the response
    response = requests.get(base_url, params=params)
    response.raise_for_status()

    # Parse the response JSON
    data = response.json()

    already_downloaded = 0
    while True:

        # First page
        if already_downloaded == 0:
            print(f"Got {data['count']} results, getting the most popular {args.num_records} books.")

        for book in data['results'][:args.num_records - already_downloaded]:
            try:
                book_url = book['formats']['text/plain; charset=us-ascii']

                if len(book['authors']) == 0:
                    author = "Unknown Author"
                else:
                    author = book['authors'][0]['name']

                download_book(book_url, args.output_dir, author + " - " + book['title'])
                already_downloaded += 1
            except KeyError:
                print(f"Skipping book {book['title']} due to error (Probably TXT format is not available.)")
            except Exception as e:
                print(f"Skipping book {book['title']} due to an unexpected error.")

        if already_downloaded >= args.num_records:
            print(f"Downloaded {already_downloaded} books.")
            break

        print(f"Current progress: {already_downloaded}/{min(data['count'], args.num_records)} downloaded")
        if args.num_records > already_downloaded and data['next'] is not None:
            print(f"Navigating to next page ({data['next']}).")
            response = requests.get(data['next'])
            response.raise_for_status()
            data = response.json()

    print(f"✅ Done.")


if __name__ == '__main__':
    main()
